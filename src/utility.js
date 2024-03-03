module.exports = class {
    // Read file, extract input and start the timestep sequence
    static extractInputAndBegin(dataset, mapping, beforeTimestep, afterTimestep) {
        const fs = require('fs');
        return new Promise((resolve, reject) => {
            const stream = fs.createReadStream(dataset);
            const map = fs.readFileSync(mapping, { encoding:'utf8' }).split('\n').slice(0, -1).reduce((acc, row) => {
                const [ index, char ] = row.split(' ');
                acc[parseInt(index)] = parseInt(char) || char;
                return acc;
            }, {});

            stream.on('data', chunk => {
                chunk.toString().split('\n').slice(1, -1).forEach(row => {
                    const data = row.split(',');
                    const char = map[parseInt(data[0])];
                    if (!char) return;
                    const input = new Map();
                    const str = String.fromCharCode(char);
                    const label = str === '\x00' ? char.trim() : str;
                    data.slice(1).forEach((pixel, index) => {
                        if (pixel === '0') return;
                        input.set(index + 1, pixel / 255);      // Input as a [0-1] representation
                    });
                    beforeTimestep(input, label, afterTimestep);
                });
            });

            stream.on('error', reject);
            stream.on('end', resolve);
        });
    }
    // Create topological inhibition square areas
    static inhibition(size, row, square, depth = 2) {
        if (!Number.isInteger(size))   throw new Error('"size" must be an integer.');
        if (!Number.isInteger(row))    throw new Error('"row" must be an integer.');
        if (!Number.isInteger(square)) throw new Error('"square" must be an integer.');
        if (!Number.isInteger(depth))  throw new Error('"depth" must be an integer.');

        const hareas = new Set();
        const vareas = new Set();
        const hareasByNodeId = new Map();
        const vareasByNodeId = new Map();
        const ycount = size / row / square;
        const xcount = row / square;

        // console.log(ycount, xcount);

        for (let y = 0; y < ycount; y++) {
            for (let x = 0; x < xcount; x++) {
                const harea = new Map();
                // console.log('H', y, x);
                for (let j = 0; j < square; j++) {
                    for (let i = 0; i < square; i++) {
                        const id = (y * square + j) * row + (x * square + i) + 1;
                        // console.log(id);
                        if (id > size) continue;
                        const varea = new Map();
                        vareas.add(varea);
                        hareas.add(harea);
                        // console.log('V');
                        for (let d = 0; d < depth; d++) {
                            const nodeId = id + size * d;
                            // console.count(nodeId);
                            hareasByNodeId.set(nodeId, harea);
                            vareasByNodeId.set(nodeId, varea);
                        }
                        if (id % row === 0) break;
                    }
                }
            }
        }

        // for (let l = 1; l <= (size * depth); l++) {
        //     console.count(!!vareasByNodeId.get(l));
        // }

        // hareasByNodeId.forEach(area => {
        //     console.log(area);
        //     !area.size && console.count("areaNoSize");
        // })

        return {
            hareas,
            hareasByNodeId
        }
    }
    // Find most active output nodes for each label
    static classify(results, kwinner) {
        results.forEach((data, label) => {
            const timesteps = data.get('timesteps');
            const nodes = new Map(data.get('nodes'));

            // Normalize weights by the timestep count
            // Each label may be active for more timesteps than others
            let sum = 0;
            nodes.forEach((value, node) => {
                sum += value / timesteps;
            });

            // Normalize weights by the sum of the weights
            // Each label representation may require different number of nodes [3 vs 8]
            nodes.forEach((value, node) => {
                nodes.set(node, value / sum);
            });

            // Sort by weight and take only top "kwinner" nodes
            const arr = this.sort(nodes);
            arr.length = arr.length > kwinner ? kwinner : arr.length;
            data.set('classifier', new Map(arr));
        });
    }
    // Score for each label determined from the classifier result
    static score(results, output) {
        const score = new Map();
        const set = new Set(output);

        results.forEach((data, label) => {
            let overall = 0;

            data.get('classifier').forEach((weight, node) => {
                overall += set.has(node) ? weight : -weight;
            });

            score.set(label, overall);
        });

        return score;
    }
    // Calculate overall score
    static overall(scores) {
        const final = new Map();

        scores.forEach(data => {
            const { correctLabel, score } = data;
            const sorted = this.sort(score);
            const bool = correctLabel === sorted[0][0]; // Is correct label at the top
            const bools = final.get(correctLabel) || new Map();
            bools.set(bool, bools.get(bool) + 1 || 1);
            final.set(correctLabel, bools);
        });

        console.log('/////////////////////////////////////// DATASET');
        console.log('/////////////////////////////////////// SCORE');

        let total = 0;
        final.forEach((bools, label) => {
            const t = bools.get(true);
            const f = bools.get(false);
            const overall = t + f;
            const value = this.toFixedFloat(t / overall) * 100 || 0;
            total += value;
            console.log(label, this.toFixedFloat(value));
        });

        console.log('*************** WARNING ***************');
        console.log('Score does not accurately describe the performance or functionality of this type of network!');
        console.log('The network learns as humans do. It observes temporal sequences and strengthens patterns that repeat many times in quick succession.');
        console.log('Therefore, the datasets should be SORTED by labels otherwise it may take longer for the network to stabilize.');
        console.log('Learning is completely unsupervised and we are only trying to understand the inner workings of the network.');
        console.log('***************************************');
        console.log('SCORE:', this.toFixedFloat(total / final.size));
    }
    // Sort nodes by weight from highest to lowest
    static sort(map) {
        return [...map].sort((a, b) => b[1] - a[1]);
    }
    // Short representation of large numbers
    static abbr(value) {
        const symbol = ['', 'k', 'M', 'G', 'T', 'P', 'E'];
        const rx = /\.0+$|(\.[0-9]*[1-9])0+$/;
        const tier = Math.log10(Math.abs(value)) / 3 | 0;
        const suffix = symbol[tier];
        if (suffix) value = value / Math.pow(10, tier * 3);
        return value.toFixed(2).replace(rx, '$1') + suffix;
    }
    // Fix float onto 2 decimal numbers
    static toFixedFloat(value) {
        return parseFloat(value.toFixed(2));
    }
    // Visualization of input and output
    static visualize(size, row, label, activations, colored) {
        console.log('======================================= LABEL ' + label);
        const log = [];
        let id = 1;

        if (colored && activations.get) {
            const colors = {
                '0.0': '\x1b[40m',
                '0.1': '\x1b[100m',
                '0.2': '\x1b[100m',
                '0.3': '\x1b[100m',
                '0.4': '\x1b[100m',
                '0.5': '\x1b[100m',
                '0.6': '\x1b[47m',
                '0.7': '\x1b[47m',
                '0.8': '\x1b[47m',
                '0.9': '\x1b[47m',
                '1.0': '\x1b[107m'
            };
            const black = colors['0.0'];

            for (; id <= size; id++) {
                const node = activations.get(id);
                log.push(node ? colors[node.toFixed(1)] : black);
                if (id % row) continue;
                console.log(log.join('  ') + ' ' + black);
                log.length = 0;
            }
        } else {
            for (; id <= size; id++) {
                log.push(activations.has(id) ? '0' : ' ');
                if (id % row) continue;
                console.log(...log);
                log.length = 0;
            }
        }
    }
    // Calculate print log for each timestep
    static log(timestep) {
        return timestep < 1e4 || !(timestep % 1e4);
    }
}
