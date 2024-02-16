const Utility = require('./utility');

module.exports = function ({
    // REQUIRED
    regions,
    // OPTIONAL
    temporalLength,
    initialLinkPermanence,
    maximumLinkPermanence,
    minimumLinksInPool,
    maximumLinksInPool,
    poolWeightChangeRate,
    initialPoolWeight,
    minimumPoolWeight,
    maximumPoolWeight,
    exponentialGrowth,
    inputMultiplier
}) {
    if (!regions) throw new Error('"regions" are not defined.');
    let Size = 0;                                           // Total number of nodes in the network
    let kwinner = 0;                                        // Number of top nodes to be picked in a classifier as a label representation
    // Layers
    const Layers = new Map();                               // All the layers in the entire network
    const NodesByLayerId = new Map();
    const AreasByLayerId = new Map();
    const AreasByLayerIdByNodeId = new Map();               // Areas mapped by nodeId for each layer
    const UnpredictedNodesByLayerId = new Map();
    const ThinkingNodesByLayerId = new Map();
    const TouchedNodesByLayerId = new Map();                // Nodes that have energy > 0 in this timestep

    // Populate layers in each region
    regions.forEach(region => {
        if (!Number.isInteger(region.size)) throw new Error('Region "size" must be an integer.');
        region.temporal = new createTemporalState();
        populateLayers(region.layers, region.size);
    });

    // Network
    let Timestep = 0;
    let TotalUpredictedNodeSizePerTimestep = 0;
    const TemporalLength = parseInt(temporalLength) || 10;                      // How many timestep input will be temporally averaged
    const InitialLinkPermanence = parseInt(initialLinkPermanence) || parseInt(Math.pow(Size, 1 / 4));
    const MaximumLinkPermanence = parseInt(maximumLinkPermanence) || parseInt(Size * Size);
    const MinimumLinksInPool = parseInt(minimumLinksInPool) || parseInt(Math.pow(Size, 1 / 8));     // Should be as low as possible while maintaining pool uniqueness
    const MaximumLinksInPool = parseInt(maximumLinksInPool) || parseInt(MinimumLinksInPool * 5);
    const PoolWeightChangeRate = parseInt(poolWeightChangeRate) || 1;
    const MinimumPoolWeight = Number.isInteger(minimumPoolWeight) ? minimumPoolWeight : -1;
    const MaximumPoolWeight = parseInt(maximumPoolWeight) || 20;
    const InitialPoolWeight = Number.isInteger(initialPoolWeight) ? initialPoolWeight : MaximumPoolWeight;
    const ExponentialGrowth = Utility.toFixedFloat(exponentialGrowth || 2);
    const InputMultiplier = Math.abs(parseInt(inputMultiplier)) || 1000;        // How many times is the driver input stronger than pool input
    const TotalPossibleLinks = parseInt(Size * Size - Size);
    const TotalPossiblePools = parseInt(TotalPossibleLinks / MinimumLinksInPool);
    const TouchedPools = new Set();     // Pools that have at least one active link, but did not activate yet
    const GlobalOutputArray = [];       // Array allows random access
 
    // Logging
    let permanentPools = 0;             // Number of pools that cannot be removed
    let liveLinks = 0;
    let createdLinks = 0;
    let deletedLinks = 0;
    let createdLinksPerTimestep = 0;
    let deletedLinksPerTimestep = 0;
    let activatedLinksPerTimestep = 0;
    let livePools = 0;
    let createdPools = 0;
    let deletedPools = 0;
    let createdPoolsPerTimestep = 0;
    let deletedPoolsPerTimestep = 0;
    let activatedPoolsPerTimestep = 0;
    let LTPPoolsPerTimestep = 0;
    let LTDPoolsPerTimestep = 0;
    let unpredictedNonCreatedPools = 0;
    let notCreatedPools = 0;
    let unpredictedSparsity = 0;
    let thinkingSparsity = 0;
    let outputSparsity = 0;
    let totalThinkingNodeSizePerTimestep = 0;
    let highestLinkPermanence = 0;

    class Layer {

        constructor(id) {
            this.id = id;
            this.Nodes = NodesByLayerId.get(this.id);
            this.Areas = AreasByLayerId.get(this.id);
            this.AreasByNodeId = AreasByLayerIdByNodeId.get(this.id);
            this.UnpredictedNodes = UnpredictedNodesByLayerId.get(this.id);
            this.ThinkingNodes = ThinkingNodesByLayerId.get(this.id);
            this.TouchedNodes = TouchedNodesByLayerId.get(this.id);
            this.Output = new Map();
            this.Threshold = 0;
        }

        run(input) {

            // Add input energy onto the nodes
            input.forEach((energy, nodeId) => {
                const node = this.Nodes.get(nodeId);
                node.energy += energy;
                this.Threshold += energy;
                this.TouchedNodes.add(node);
            });

            // Calculate dynamic energy threshold for each layer
            this.Threshold /= this.TouchedNodes.size;

            // Check all nodes that have energy and calculate each area output nodes
            this.TouchedNodes.forEach(node => {
                if (this.Threshold > node.energy) return;
                const area = this.AreasByNodeId.get(node.id);
                const highest = area.get(0);

                if (node.input.size) {
                    if (highest) {
                        if (node.energy > highest.energy) area.set(0, node);
                    } else {
                        area.clear();
                        area.set(0, node);
                    }
                } else {
                    if (highest) return;
                    area.set(node.id, node);
                }
            });

            this.Threshold = 0;
            this.UnpredictedNodes.clear();
            this.ThinkingNodes.clear();

            // Set upredicted nodes and reward correctly predicting pools
            this.Areas.forEach(area => {
                area.forEach(node => {
                    if (node.input.size) {
                        if (!input.has(node.id)) this.ThinkingNodes.add(node.id);
                    } else {
                        this.UnpredictedNodes.set(node.id, node);
                    }
                    node.reward();
                    this.TouchedNodes.delete(node);
                });
            });

            TotalUpredictedNodeSizePerTimestep += this.UnpredictedNodes.size;
            totalThinkingNodeSizePerTimestep += this.ThinkingNodes.size;

            // Punish and clear all remaining incorrectly predicting pools
            this.TouchedNodes.forEach(node => node.punish());
            this.TouchedNodes.clear();
        }

        activate() {
            this.Output.clear();
            // Create output and activate output nodes
            this.Areas.forEach(area => {
                area.forEach(node => {
                    GlobalOutputArray.push(node);
                    this.Output.set(node.id, 1);
                    node.activate(this);
                });
                area.clear();
            });
        }

        log() {
            console.log('================ NODES ================ LAYER ' + this.id);
            console.log('           Output ->', this.Output.size + ' - ' + [...this.Output.keys()].sort());
            console.log('      Unpredicted ->', this.UnpredictedNodes.size || this.UnpredictedNodes.size.toString(), this.UnpredictedNodes.size ? '- ' + [...this.UnpredictedNodes.keys()].sort() : '');
            // Unpredicted -> Nodes that became active in this timestep but were not predicted by any of the pools
            console.log('         Thinking ->', this.ThinkingNodes.size || this.ThinkingNodes.size.toString(), this.ThinkingNodes.size ? '- ' + [...this.ThinkingNodes.keys()].sort() : '');
            // Thinking -> Nodes that became active in this timestep without direct input, instead their energy came only from the pools
            console.log('  Output/Timestep ->', Utility.toFixedFloat(this.Output.size / this.Nodes.size * 100) + '%');
        }
    }

    class Link {

        constructor() {
            this.permanence = InitialLinkPermanence;
            this.decayed = Timestep;                        // Last timestep it was decayed
        }

        reward(pool) {
            if (this.permanence > MaximumLinkPermanence) {
                this.permanence = Infinity;
                pool.permanent.add(this);
            } else {
                this.permanence *= ExponentialGrowth;       // Exponential growth of links involved in correct prediction
                if (this.permanence > highestLinkPermanence) highestLinkPermanence = this.permanence;
            }
        }

        decay() {
            this.permanence -= Timestep - this.decayed;     // Decay link permanence depending on the last time it was decayed
            if (this.permanence > 0) {
                this.decayed = Timestep;
                return false;
            }
            return true;
        }
    }

    class Pool {

        /*
            Pool class is used for managing links and their permanences.
            There is no way of knowing in advance which links work well together.
            Over time pool will settle on only the links that activate together.
            Those links will eventually achieve MaximumLinkPermanence value.
            When that happens, Pool and Links inside are no longer needed.
            Finally, Pool class is replaced by PermanentPool class.
        */

        constructor(output) {
            this.input = new Map();                     // Map(Node, Link)
            this.output = output;                       // Output node
            this.weight = InitialPoolWeight;            // Pool weight starts with a InitialPoolWeight
            this.predicting = new Set();                // Links that are predicting in this timestep
            this.permanent = new Set();                 // Links that have reached MaximumLinkPermanence
            this.activated = false;                     // If it was activated in this timestep
        }

        activate(node) {
            const link = this.input.get(node);
            if (link.decay()) return this.decay(node);  // Decay the link, if it was removed then decay the pool
            // ++activatedLinksPerTimestep;
            this.predicting.add(link);                  // Keep adding links for reward even if pool was activated
            if (this.activated) return;                 // Prevent activation of the pool if it is was already active in this timestep
            TouchedPools.add(this);                     // Pool now has at least one active link so it is added into TouchedPools
            if (MinimumLinksInPool > this.predicting.size) return;
            this.activated = true;
            this.output.touch(this);
            TouchedPools.delete(this);                  // Pool has been activated so it is removed from TouchedPools
            // ++activatedPoolsPerTimestep;
        }

        reward() {
            if (this.weight < MaximumPoolWeight) this.weight += PoolWeightChangeRate;   // Increase pool weight
            this.predicting.forEach(link => link.reward(this));                         // Reward all the links that were active in previous timestep
            if (this.permanent.size === this.input.size) this.replace();                // Replace only after all the links in pool have reached MaximumLinkPermanence
            this.predicting.clear();
            this.activated = false;
            // ++LTPPoolsPerTimestep;
        }

        punish() {
            if (this.weight > MinimumPoolWeight) this.weight -= PoolWeightChangeRate;   // Decrease pool weight
            this.predicting.clear();
            this.activated = false;
            // ++LTDPoolsPerTimestep;
        }

        decay(node) {
            this.input.delete(node);                                                    // Delete link from pool input
            node.output.delete(this.output);                                            // Delete the pool from its input node
            ++deletedLinksPerTimestep;
            if (MinimumLinksInPool > this.input.size) {                                 // Check if pool has enough links to activate the pool
                ++deletedPoolsPerTimestep;                  
                deletedLinksPerTimestep += this.input.size;
                this.input.forEach((link, node) => node.output.delete(this.output));    // Delete the pool from every node output
                this.output.input.delete(this);                                         // Delete the pool from output node input
                this.predicting.clear();
                this.permanent.clear();
                this.input.clear();
            }
        }

        replace() {
            const pool = new PermanentPool(this.output, this.weight);
            this.input.forEach((link, node) => node.output.set(this.output, pool));     // Populate every node output with new PermanentPool
            this.output.input.delete(this);
            this.permanent.clear();
            this.input.clear();
            ++permanentPools;
        }

        clear() {
            this.predicting.clear();
        }
    }

    class PermanentPool {

        /*
            Linear decay is the only force in the network that can lower link permanence.
            In the network with 1e6 nodes, MaximumLinkPermanence will be 1e12.
            If we have 100 timesteps per second and a linear decay of -1 per timestep, it would take +300 real life years for permanence of 1e12 to decay to 0.
            Therefore, when every single link in the pool reaches MaximumLinkPermanence value then that pool can be considered permanent.
            We replace Pool class with new PermanentPool class.
            PermanentPool class is functionaly identical to Pool class but is optimized for high performance.
            This implementation detail is not essential but it does have a significant performance impact.
        */

        constructor(output, weight) {
            this.output = output;                   // Output node
            this.weight = weight;
            this.predicting = 0;                    // Number of nodes predicting
            this.activated = false;                 // If it was activated in this timestep
        }

        activate() {
            // ++activatedLinksPerTimestep;
            if (this.activated) return;             // Prevent activation of the pool if it is was already active in this timestep
            ++this.predicting;
            TouchedPools.add(this);                 // Pool now has at least one active link so it is added into TouchedPools
            if (MinimumLinksInPool > this.predicting) return;
            this.activated = true;
            this.output.touch(this);
            TouchedPools.delete(this);              // Pool has been activated so it is removed from TouchedPools
            // ++activatedPoolsPerTimestep;
        }

        reward() {
            if (this.weight < MaximumPoolWeight) this.weight += PoolWeightChangeRate;   // Increase pool weight
            this.activated = false;
            this.predicting = 0;
            // ++LTPPoolsPerTimestep;
        }

        punish() {
            if (this.weight > MinimumPoolWeight) this.weight -= PoolWeightChangeRate;   // Decrease pool weight
            this.activated = false;
            this.predicting = 0;
            // ++LTDPoolsPerTimestep;
        }

        clear() {
            this.predicting = 0;
        }
    }

    class Node {

        constructor(id, layer) {
            this.id = id;
            this.layer = layer;                     // Layer instance that this node belongs to
            this.input = new Set();                 // Pools that were predicting in previous timestep
            this.output = new Map();                // Map(Node, Pool) - Pools that this node has to other nodes
            this.energy = 0;                        // Energy in this timestep
        }

        activate() {
            this.output.forEach(pool => pool.activate(this));
        }

        touch(pool) {
            this.input.add(pool);                   // Add pool into the input of this node
            this.energy += pool.weight;             // Increase the energy of this node
            this.layer.Threshold += pool.weight;    // Add into the overall sum of the layer energy
            this.layer.TouchedNodes.add(this);      // Node now has energy so it is added into its layer TouchedNodes
        }

        reward() {
            this.input.forEach(pool => pool.reward());
            this.input.clear();
            this.energy = 0;
        }

        punish() {
            this.input.forEach(pool => pool.punish());
            this.input.clear();
            this.energy = 0;
        }
    }

    // Create layers
    regions.forEach(region => createLayers(region.layers));

    // Create nodes for each layer
    AreasByLayerIdByNodeId.forEach((areasByNodeId, layerId) => {
        areasByNodeId.forEach((area, nodeId) => {
            NodesByLayerId.get(layerId).set(nodeId, new Node(nodeId, Layers.get(layerId)))
        });
    });

    // Log network stats
    function stats() {
        console.log('/////////////////////////////////////// NETWORK');
        console.log('/////////////////////////////////////// STATS');
        Layers.forEach(layer => console.log('                     LAYER -> ' + 'ID:' + layer.id + '  NODES:' + layer.Nodes.size + '  AREAS:' + layer.Areas.size));
        console.log('     TOTAL NUMBER OF NODES -> ' + Size);
        console.log('       INITIAL POOL WEIGHT -> ' + InitialPoolWeight);
        console.log('       MINIMUM POOL WEIGHT -> ' + MinimumPoolWeight);
        console.log('       MAXIMUM POOL WEIGHT -> ' + MaximumPoolWeight);
        console.log('   POOL WEIGHT CHANGE RATE -> ' + PoolWeightChangeRate);
        console.log('   INITIAL LINK PERMANENCE -> ' + InitialLinkPermanence);
        console.log('   MAXIMUM LINK PERMANENCE -> ' + MaximumLinkPermanence);
        console.log('     MINIMUM LINKS IN POOL -> ' + MinimumLinksInPool);
        console.log('     MAXIMUM LINKS IN POOL -> ' + MaximumLinksInPool);
        console.log('          INPUT MULTIPLIER -> ' + InputMultiplier);
        console.log('              LINEAR DECAY -> ' + -1);
        console.log('        EXPONENTIAL GROWTH -> ' + ExponentialGrowth);
        console.log('           TEMPORAL LENGTH -> ' + TemporalLength);
        console.log('                   KWINNER -> ' + kwinner);
    }

    // Temporal state used to calculate temporal moving average
    function createTemporalState() {
        this.stack = [];            // Last "length" inputs
        this.current = new Map();   // Last average calculation
    }

    // Generate input moving average
    function generateTemporalInput(state, input, length, multiplier) {
        const stack = state.stack;
        const current = state.current;
        const steps = stack.length || 1;
        const next = new Map();

        stack.unshift(new Map(input));     // Add infront

        input.forEach((value, nodeId) => {
            current.set(nodeId, current.get(nodeId) + value || value);
        });

        stack.length > length && stack.pop().forEach((value, nodeId) => {     // Remove last
            const weight = current.get(nodeId) - value;
            weight > 0.01 ? current.set(nodeId, weight) : current.delete(nodeId);
        });

        current.forEach((value, nodeId) => {
            next.set(nodeId, (value / steps) * multiplier);
        });

        return next;    // In this way, performance of the function remains O(1) no matter the temporal length
    }

    // Populate layer structure with initial params
    function populateLayers(layers, size) {
        layers.forEach(layer => {
            if (!layer.id)                    throw new Error('Layer "id" is not defined.');
            if (NodesByLayerId.has(layer.id)) throw new Error('Layer id "' + layer.id + '" is not unique.');
            Size += size;
            NodesByLayerId.set(layer.id, new Map());
            TouchedNodesByLayerId.set(layer.id, new Set());
            ThinkingNodesByLayerId.set(layer.id, new Set());
            UnpredictedNodesByLayerId.set(layer.id, new Map());
            const inhibition = Utility.inhibition(size, layer.inhibition.row, layer.inhibition.square);
            AreasByLayerId.set(layer.id, inhibition.areas);
            AreasByLayerIdByNodeId.set(layer.id, inhibition.areasByNodeId);
            kwinner += inhibition.areas.size;
            if (layer.layers) {
                layer.temporal = new createTemporalState();
                populateLayers(layer.layers, size);
            }
        });
    }

    // Create layer instances
    function createLayers(layers) {
        layers.forEach(layer => {
            layer.instance = new Layer(layer.id);
            Layers.set(layer.id, layer.instance);
            layer.layers && createLayers(layer.layers);
        });
    }

    // Run all layers in all regions
    function runLayers(layers, temporal, input) {
        const current = generateTemporalInput(temporal, input, TemporalLength, InputMultiplier);    // Temporal input is shared among first children
        layers.forEach(layer => {
            layer.instance.run(current);
            layer.layers && runLayers(layer.layers, layer.temporal, layer.instance.Output);
        });
    }

    // Run single timestep
    function timestep(inputs, label) {
        ++Timestep;

        // Run the layers as a recursive feedforward input
        regions.forEach((region, index) => runLayers(region.layers, region.temporal, inputs[index]));

        // Clear all inactive pools
        TouchedPools.forEach(pool => pool.clear());
        TouchedPools.clear();
 
        // Link previous timestep output nodes onto the unpredicted nodes
        if (TotalUpredictedNodeSizePerTimestep) {
            const outputLength = GlobalOutputArray.length;                                  // GlobalOutputArray contains output nodes from previous timestep
            const random = new Set();                                                       // Set enforces uniqueness of elements - there is a possibility of randomly picking the same node

            for (let i = 0; i < MaximumLinksInPool; i++) {                                  // Pick random node from global output from previous timestep
                random.add(GlobalOutputArray[Math.floor(Math.random() * outputLength)]);    // Array allows random access
            }

            if (random.size >= MinimumLinksInPool) {
                Layers.forEach(layer => {
                    layer.UnpredictedNodes.forEach(output => {
                        const linking = [];                                                 // Prepare the nodes that will be used for linking
                        random.forEach(node => {
                            if (node === output || node.output.has(output)) return;         // Node cannot link to itself | Allow only single link between nodes
                            linking.push(node);
                        });

                        if (linking.length < MinimumLinksInPool) return;

                        const pool = new Pool(output);                                      // Create just one pool per output node per timestep
                        linking.forEach(node => {
                            pool.input.set(node, new Link());                               // Populate pool input with Links
                            node.output.set(output, pool);                                  // Populate every output node with this pool
                        });
                        ++createdPoolsPerTimestep;
                        createdLinksPerTimestep += pool.input.size;
                    });
                });
            }
        }

        // Clear global output
        GlobalOutputArray.length = 0;

        // Activate the network
        Layers.forEach(layer => layer.activate());

        createdPools += createdPoolsPerTimestep;
        deletedPools += deletedPoolsPerTimestep;
        !createdPoolsPerTimestep && ++notCreatedPools;
        createdPoolsPerTimestep < TotalUpredictedNodeSizePerTimestep && ++unpredictedNonCreatedPools;
        createdLinks += createdLinksPerTimestep;
        deletedLinks += deletedLinksPerTimestep;
        liveLinks = createdLinks - deletedLinks;
        livePools = createdPools - deletedPools;
        outputSparsity += Utility.toFixedFloat(GlobalOutputArray.length / Size);
        thinkingSparsity += Utility.toFixedFloat(totalThinkingNodeSizePerTimestep / GlobalOutputArray.length);
        unpredictedSparsity += Utility.toFixedFloat(TotalUpredictedNodeSizePerTimestep / GlobalOutputArray.length);
        if (Utility.log(Timestep) || totalThinkingNodeSizePerTimestep) {
            console.log('/////////////////////////////////////// TIMESTEP ' + Timestep);
            console.log('/////////////////////////////////////// LABEL ' + label);
            console.log('================ NETWORK ==============');
            console.log('   Output/Average ->', Utility.toFixedFloat(outputSparsity / Timestep * 100) + '%');
            // Output/Average -> Average output percentage across all timesteps and all nodes
            console.log('   Unpred/Average ->', Utility.toFixedFloat(unpredictedSparsity / Timestep * 100) + '%');
            // Unpred/Average -> Percentage of all the timesteps where there were upredicted nodes
            // The lower the better
            console.log(' Thinking/Average ->', Utility.toFixedFloat(thinkingSparsity / Timestep * 100) + '%');
            console.log(' Ncreated/Average ->', Utility.toFixedFloat(notCreatedPools / Timestep * 100) + '%');
            // Ncreated/Average -> Percentage of all the timesteps where no new pools created have been created
            // The higher the better
            console.log('  Ncreated/Unpred ->', Utility.abbr(unpredictedNonCreatedPools) + ' - ' + Utility.toFixedFloat(unpredictedNonCreatedPools / Timestep * 100) + '%');
            // Ncreated/Unpred -> How many timesteps there were unpredicted nodes but there were no new pools created
            // Node can link to any another node only once (This rule could be changed)
            // If the node is unpredicted, but all of the output nodes from previous timestep are already connected to it in different pools
            // Then, even tho there was unpredicted node, new pool has not been created and this results in missed learning
            // Usually, this happens if the number of output nodes, determined by inhibition areas, is too small compared to the number of MinimumLinksInPool
            // Either reduce the inhibition square area to increase the number of output nodes and/or decrease MinimumLinksInPool and/or increase MaximumLinksInPool
            // Except from the very first timestep, the value should remain 1
            Layers.forEach(layer => layer.log());
            console.log('================ LINKS ================');
            console.log('             Live ->', Utility.abbr(liveLinks));
            console.log('          Deleted ->', Utility.abbr(deletedLinks) + ' - ' + Utility.toFixedFloat(deletedLinks / createdLinks * 100) + '%');
            console.log(' Created/Timestep ->', createdLinksPerTimestep || createdLinksPerTimestep.toString());
            console.log(' Deleted/Timestep ->', deletedLinksPerTimestep || deletedLinksPerTimestep.toString());
            // console.log('  Active/Timestep ->', Utility.abbr(activatedLinksPerTimestep) + ' - ' + Utility.toFixedFloat(activatedLinksPerTimestep / liveLinks * 100) + '%');
            console.log('       Links/Pool ->', Utility.toFixedFloat(liveLinks / livePools).toString());
            console.log('       Links/Node ->', Utility.toFixedFloat(liveLinks / Size).toString());
            console.log('         Sparsity ->', Utility.toFixedFloat(liveLinks / TotalPossibleLinks * 100) + '%');
            // Sparsity -> Percentage of all the possible links that could ever be created
            console.log('HighestPermanence ->', highestLinkPermanence >= MaximumLinkPermanence ? 'Maximum' : Math.floor(highestLinkPermanence));
            // HighestPermanence -> Highest permanence that any of the links currently have
            // Useful when changing the difficulty for increasing link permanence
            // Difficuly is too high if non of the links could reach the MaximumLinkPermanence
            console.log('================ POOLS ================');
            console.log('             Live ->', Utility.abbr(livePools));
            console.log('          Deleted ->', Utility.abbr(deletedPools) + ' - ' + Utility.toFixedFloat(deletedPools / createdPools * 100) + '%');
            console.log(' Created/Timestep ->', createdPoolsPerTimestep || createdPoolsPerTimestep.toString());
            console.log(' Deleted/Timestep ->', deletedPoolsPerTimestep || deletedPoolsPerTimestep.toString());
            // console.log('  Active/Timestep ->', Utility.abbr(activatedPoolsPerTimestep) + ' - ' + Utility.toFixedFloat(activatedPoolsPerTimestep / livePools * 100) + '%');
            // console.log('     LTP/Timestep ->', Utility.abbr(LTPPoolsPerTimestep));
            // console.log('     LTD/Timestep ->', Utility.abbr(LTDPoolsPerTimestep));
            console.log('       Pools/Node ->', Utility.toFixedFloat(livePools / Size).toString());
            console.log('         Sparsity ->', Utility.toFixedFloat(livePools / TotalPossiblePools * 100) + '%');
            // Sparsity -> Percentage of all the possible pools that could ever be created
            console.log('   PermanentPools ->', Utility.abbr(permanentPools) + ' - ' + Utility.toFixedFloat(permanentPools / livePools * 100) + '%');
            // PermanentPools -> Pools that will never be removed
            // Only when all of the links in the pool reach MaximumLinkPermanence then the Pool class gets replaced by PermanentPool class
            // This implementation detail is for performance reasons only
            // Number of permanent pools should grow continuously
        }
        LTPPoolsPerTimestep = 0;
        LTDPoolsPerTimestep = 0;
        createdPoolsPerTimestep = 0;
        deletedPoolsPerTimestep = 0;
        activatedPoolsPerTimestep = 0;
        createdLinksPerTimestep = 0;
        activatedLinksPerTimestep = 0;
        deletedLinksPerTimestep = 0;
        totalThinkingNodeSizePerTimestep = 0;
        TotalUpredictedNodeSizePerTimestep = 0;

        return GlobalOutputArray;
    }

    return {
        stats,
        timestep,
        kwinner
    }
}
