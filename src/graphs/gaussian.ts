import { GraphType } from './util';
import { matrix, multiply } from 'mathjs';
import type { FactorGraph, VariableNode, FactorNode, InternalFactorGraph, NodeID, GraphNodeInternal } from './util';

export interface GaussianVariableInput extends VariableNode {
    mean: number;
    variance: number;
}

export interface GaussianVariable extends GaussianVariableInput, GraphNodeInternal {
    neighbors: GaussianFactor[];
    currentMean: number;
    currentVariance: number;
}

export interface GaussianFactorInput extends FactorNode {
    potentialMean: number[];
    potentialPrecision: number[][];
}

export interface GaussianFactor extends GaussianFactorInput, GraphNodeInternal {
    neighbors: GaussianVariable[];
}

export type GaussianNodeInput = GaussianVariableInput | GaussianFactorInput;
export type GaussianNode = GaussianVariable | GaussianFactor;

export type GaussianGraph = InternalFactorGraph<GaussianVariable, GaussianFactor>

export class GaussianBeliefPropagation {
    public graph: GaussianGraph;
    private messages: {
        mean: number[][];
        precision: number[][];
    };
    private nextMessages: {
        mean: number[][];
        precision: number[][];
    };
    private dampingFactor: number;
    private evidence = new Map<GaussianVariable, { mean: number, variance: number }>();
    
    constructor(graph: FactorGraph<GaussianNodeInput>, options: { damping?: number } = {}) {
        this.graph = this.processGraph(graph);
        this.dampingFactor = options.damping || 0;
        this.messages = {
            mean: [],
            precision: []
        };
        this.nextMessages = {
            mean: [],
            precision: []
        };
        this.evidence = new Map<GaussianVariable, { mean: number, variance: number }>();
        this.initializeMessages();
    }

    private processGraph(graph: FactorGraph<GaussianNodeInput>): GaussianGraph {
        let nodes: [NodeID, GaussianNode][] = Object.entries(graph.nodes).map(([nodeId, node], i) => (
            node.type === GraphType.VARIABLE ?
                [nodeId, Object.assign(
                    {},
                    node,
                    {name: nodeId, id: i, neighbors: [], currentMean: node.mean, currentVariance: node.variance}
                ) as GaussianVariable]:
                [nodeId, Object.assign(
                    {},
                    node,
                    {name: nodeId, id: i, neighbors: []}
                ) as GaussianFactor]
        ));

        const nodeDict = Object.fromEntries(nodes);

        const edges = graph.edges
            .filter(
                ([from, to]) => from in nodeDict && to in nodeDict
            )
            .map(
                ([from, to]) => [nodeDict[from]?.id || 0, nodeDict[to]?.id || 0] as [number, number]
            )
            .flatMap(([from, to]) => [[from, to], [to, from]] as [number, number][]);

        let nodesOut = nodes.map(([_, o]) => o);

        edges.forEach(([from, to]) => {
            nodesOut[from]?.type == GraphType.VARIABLE && nodesOut[to] && nodesOut[from]?.type != nodesOut[to]?.type ?
            nodesOut[from].neighbors.push(nodesOut[to]) :
            nodesOut[from]?.type == GraphType.FACTOR && nodesOut[to] && nodesOut[from]?.type != nodesOut[to]?.type ?
            nodesOut[from].neighbors.push(nodesOut[to]) :
                console.error(`Invalid edge: (${from}, ${to})`);
        });

        return { nodes: nodes.map(([_, o]) => o), edges };
    }

    private initializeMessages(): void {
        // Initialize message arrays
        const nodeCount = this.graph.nodes.length;
        this.messages.mean = Array(nodeCount).fill(0).map(() => Array(nodeCount).fill(0));
        this.messages.precision = Array(nodeCount).fill(0).map(() => Array(nodeCount).fill(0));
        this.nextMessages.mean = Array(nodeCount).fill(0).map(() => Array(nodeCount).fill(0));
        this.nextMessages.precision = Array(nodeCount).fill(0).map(() => Array(nodeCount).fill(0));

        // Initialize messages from variables to factors
        for (const node of this.graph.nodes) {
            if (node.type === GraphType.VARIABLE) {
                for (const factor of node.neighbors) {
                    this.messages.mean[node.id][factor.id] = node.mean;
                    this.messages.precision[node.id][factor.id] = 1 / node.variance;
                }
            }
        }
    }
    
    private updateVariableToFactorMessage(variable: GaussianVariable, factor: GaussianFactor): void {
        if (this.evidence.has(variable)) {
            const evidence = this.evidence.get(variable)!;
            this.nextMessages.mean[variable.id][factor.id] = evidence.mean;
            this.nextMessages.precision[variable.id][factor.id] = 1 / evidence.variance;
            return;
        }

        let otherNeighbors = variable.neighbors.filter(n => n.id != factor.id)

        let totalPrecision = otherNeighbors
            .map(n => this.messages.precision[n.id][variable.id])
            .reduce((p, c) => p + c);

        let weightedMean = otherNeighbors
            .map(n => this.messages.precision[n.id][variable.id] * this.messages.mean[n.id][variable.id])
            .reduce((p, c) => p + c);

        this.nextMessages.precision[variable.id][factor.id] = totalPrecision;
        this.nextMessages.mean[variable.id][factor.id] = totalPrecision > 0 ? weightedMean / totalPrecision : 0;
    }

    private updateFactorToVariableMessage(factor: GaussianFactor, variable: GaussianVariable): void {
        const idx = factor.neighbors.findIndex(v => v.id === variable.id);
        const n = factor.neighbors.length;
        const Lambda = matrix(factor.potentialPrecision);
        const eta = multiply(Lambda, matrix(factor.potentialMean));

        let precSum = 0;

        for (let j = 0; j < n; j++) {
            if (j === idx) continue;
            precSum += Lambda.get([idx, j])
                * this.messages.mean[factor.neighbors[j].id][factor.id];
        }

        const precision = Lambda.get([idx, idx]);
        const meanNumerator = eta.get([idx]) - precSum;

        this.nextMessages.precision[factor.id][variable.id] = precision;
        this.nextMessages.mean[factor.id][variable.id] = precision > 0 ? meanNumerator / precision : 0;
    }
    

    private updateAllMessages(): void {
        // Update variable to factor messages
        for (const node of this.graph.nodes) {
            if (node.type === GraphType.VARIABLE) {
                for (const factor of node.neighbors) {
                    this.updateVariableToFactorMessage(node, factor);
                }
            }
        }

        // Update factor to variable messages
        for (const node of this.graph.nodes) {
            if (node.type === GraphType.FACTOR) {
                for (const variable of node.neighbors) {
                    this.updateFactorToVariableMessage(node, variable);
                }
            }
        }

        // Apply damping
        for (let i = 0; i < this.graph.nodes.length; i++) {
            for (let j = 0; j < this.graph.nodes.length; j++) {
                if (!isNaN(this.nextMessages.mean[i][j])) {
                    this.nextMessages.mean[i][j] =
                        this.dampingFactor * this.messages.mean[i][j] +
                        (1 - this.dampingFactor) * this.nextMessages.mean[i][j];

                    this.nextMessages.precision[i][j] = 
                        this.dampingFactor * this.messages.precision[i][j] + 
                        (1 - this.dampingFactor) * this.nextMessages.precision[i][j];
                }
            }
        }

        // Swap messages for next iteration
        [this.messages, this.nextMessages] = [this.nextMessages, this.messages];
    }

    private updateBeliefs(): void {
        for (const node of this.graph.nodes) {
            if (node.type === GraphType.VARIABLE) {
                // Skip belief updates for observed nodes
                if (this.evidence.has(node)) continue;

                // Sum all incoming precisions and weighted means
                let totalPrecision = 0;
                let weightedMean = 0;
                
                for (const neighbor of node.neighbors) {
                    const precision = this.messages.precision[neighbor.id][node.id];
                    totalPrecision += precision;
                    weightedMean += precision * this.messages.mean[neighbor.id][node.id];
                }

                // Update beliefs
                if (totalPrecision > 0) {
                    node.currentVariance = 1 / (totalPrecision + 1e-10);
                    node.currentMean = weightedMean / (totalPrecision + 1e-10);
                } else {
                    // If no information, revert to prior
                    node.currentVariance = node.variance;
                    node.currentMean = node.mean;
                }
            }
        }
    }

    public runIterations(maxIterations: number, tolerance: number = 1e-6): void {
        for (let i = 0; i < maxIterations; i++) {
            const oldMeans = this.graph.nodes
                .filter(node => node.type === GraphType.VARIABLE)
                .map(node => node.currentMean);
            
            const oldVariances = this.graph.nodes
                .filter(node => node.type === GraphType.VARIABLE)
                .map(node => node.currentVariance);

            this.updateAllMessages();
            this.updateBeliefs();

            // Check convergence by comparing mean and variance changes
            let maxDiff = 0;
            let varIndex = 0;
            
            for (const node of this.graph.nodes) {
                if (node.type === GraphType.VARIABLE) {
                    const meanDiff = Math.abs(node.currentMean - oldMeans[varIndex]);
                    const varianceDiff = Math.abs(node.currentVariance - oldVariances[varIndex]);
                    maxDiff = Math.max(maxDiff, meanDiff, varianceDiff);
                    varIndex++;
                }
            }

            if (maxDiff < tolerance) {
                console.log(`Converged after ${i + 1} iterations`);
                return;
            }
        }
        console.log(`Did not converge after ${maxIterations} iterations`);
    }

    /**
     * Sets evidence by clamping a variable to a specific Gaussian distribution
     * @param nodeId - Name of the variable node
     * @param mean - The observed mean
     * @param variance - The observed variance
     */
    public setEvidence(nodeId: string, mean: number, variance: number): void {
        const node = this.graph.nodes.find(n => n.name === nodeId);
        if (!node || node.type !== GraphType.VARIABLE) return;

        // Store evidence
        this.evidence.set(node, { mean, variance });

        // Force beliefs to observed values
        node.currentMean = mean;
        node.currentVariance = variance;

        // Initialize messages to reflect evidence
        for (const factor of node.neighbors) {
            this.messages.mean[node.id][factor.id] = mean;
            this.messages.precision[node.id][factor.id] = 1 / variance;
            this.nextMessages.mean[node.id][factor.id] = mean;
            this.nextMessages.precision[node.id][factor.id] = 1 / variance;
        }
    }

    /**
     * Gets the current belief distribution for a node
     * @param nodeId - Name of the node
     * @returns Object with mean and variance of the Gaussian belief
     */
    public getBeliefs(nodeId: string): { mean: number, variance: number } {
        const node = this.graph.nodes.find(n => n.name === nodeId);
        if (!node || node.type !== GraphType.VARIABLE) {
            throw new Error(`Cannot get beliefs from non-variable node: ${nodeId}`);
        }
        return { mean: node.currentMean, variance: node.currentVariance };
    }
}