const Infinity = require('./infinity');
const Utility = require('./utility');

async function training(datasource, beforeTimestep) {
    const results = new Map();

    function afterTimestep(output, label) {
        const data = results.get(label) || new Map();
        const nodes = data.get('nodes') || new Map();
        output.forEach(node => nodes.set(node, nodes.get(node) + 1 || 1));
        data.set('timesteps', data.get('timesteps') + 1 || 1);
        data.set('nodes', nodes);
        results.set(label, data);
    }

    console.time('TIME-TRAINING');
    for (var i = 0; i < 1; i++) {
        await Utility.extractInputAndBegin(
            datasource + '-train-SORTED.csv',
            datasource + '-mapping.txt',
            beforeTimestep,
            afterTimestep
        );
    }
    console.timeEnd('TIME-TRAINING');

    return results;
}

async function testing(datasource, beforeTimestep, results) {
    const scores = new Set();

    function afterTimestep(output, label) {
        scores.add({
            correctLabel: label,
            score: Utility.score(results, output)
        });
    }

    console.time('TIME-TESTING');
    await Utility.extractInputAndBegin(
        datasource + '-test-SORTED.csv',
        datasource + '-mapping.txt',
        beforeTimestep,
        afterTimestep
    );
    console.timeEnd('TIME-TESTING');

    return scores;
}

async function run() {
    /*
        Regions and layers allow us to organize the network based on different inputs and input sizes.
        Network has to have at least one region with one layer. Every region in the network will receive its own main feedforward input.
        Each region consists of the layer structure, which can be nested horizontally and/or vertically.
        Horizontally added layers share the same temporally pooled input.
        Region defines a node "size" which will be used to create a fixed number of nodes for each layer inside.
        Layers inside the regions allow us to set specific inhibition rules.
        Different inhibition rules enable different ways of finding patterns.
        
        regions: [{
            size: total number of nodes in this region
            layers: [{
                id: layer id
                inhibition: {
                    row: number of nodes in each row of the layer
                    square: number of nodes in one side of the square
                },
                layers: [...]
            }]
        }]

        Network can have many regions and layers and they are usually created for the following reasons:

        Regions:
        - Different modalities - vision and hearing will have different inputs and/or different input size

        Layers:
        - Spatial patterns - same input is passed into two or more horizontal layers with a different "square" area (smaller square areas capture more details while larger square areas observe larger patterns)
        - Temporal patterns - output of the parent layer is used to create temporal input which is then passed into the child layer
    
        It is important to keep in mind that, from the standpoint of the network, regions and layers do not exist!
        Regardless of the number of regions and layers, network is a single unified system.
        In each timestep, nodes will be picked across the entire network and linked into pools.
    */
    const regions = [{
        size: 784,                  // Size is the same for all layers in this region
        depth: 1,
        layers: [{                  // First feedforward layer
            id: 'A2',               // Id has to be unique among layers
            inhibition: {
                row: 28,
                square: 2
            },
            // layers: [{           // Feedforward nested layer
            //     id: '3',         // It receives input as temporal output from its parent layer
            //     inhibition: {
            //         row: 28,
            //         square: 3
            //     },
            //     layers: [{
            //         id: '4',
            //         inhibition: {
            //             row: 28,
            //             square: 4
            //         },
            //         layers: [{
            //             id: '7',
            //             inhibition: {
            //                 row: 28,
            //                 square: 7
            //             },
            //         }]
            //     }]
            // }]
        }
        // , {                         
        //     id: 'B7',            // Temporal state is shared among each layer on horizontal level
        //     inhibition: {
        //         row: 28,
        //         square: 7
        //     },
        //     layers: [{
        //         id: 'B4',
        //         inhibition: {
        //             row: 28,
        //             square: 4
        //         },
        //     }]
        // }, {
        //     id: 'C4',
        //     inhibition: {
        //         row: 28,
        //         square: 4
        //     }
        // }, {
        //     id: 'D7',
        //     inhibition: {
        //         row: 28,
        //         square: 7
        //     }
        // }
        ]
    }];

    const network = Infinity({
        // REQUIRED
        regions
        // OPTIONAL
        // temporalLength
        // initialLinkPermanence
        // maximumLinkPermanence
        // minimumLinksInPool
        // maximumLinksInPool
        // poolWeightChangeRate
        // initialPoolWeight
        // minimumPoolWeight
        // maximumPoolWeight
        // exponentialGrowth
        // inputMultiplier
    });

    function beforeTimestep(input, label, afterTimestep) {
        const output = network.timestep([ input ], label);
        // Utility.visualize(784, 28, label, input, true);
        // Utility.visualize(size, row, label, output);
        afterTimestep(output, label);
    }

    const datasource = '../datasets/mnist';
    // const datasource = '../datasets/emnist-mnist';
    // const datasource = '../datasets/emnist-fashion';
    // const datasource = '../datasets/emnist-letters';
    // const datasource = '../datasets/emnist-balanced';

    console.time('TIME-TOTAL');
    network.stats();
    const results = await training(datasource, beforeTimestep);
    Utility.classify(results, network.kwinner);
    const scores = await testing(datasource, beforeTimestep, results);
    network.stats();
    Utility.overall(scores);
    console.timeEnd('TIME-TOTAL');
}

run();
