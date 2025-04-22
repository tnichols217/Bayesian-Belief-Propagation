import { BeliefUtils, GraphType } from './util';
import type { FactorGraph, VariableNode, FactorNode, InternalFactorGraph, NodeID, GraphNodeInternal } from './util';

export interface BayesVariableInput extends VariableNode {
    possibleValues: number[];
    currentBelief: number[];
}

export interface BayesVariable extends BayesVariableInput, GraphNodeInternal {
    neighbors: BayesFactor[];
}

export interface BayesFactorInput extends FactorNode {
    potential: (values: number[]) => number;
}

export interface BayesFactor extends BayesFactorInput, GraphNodeInternal {
    currentBelief: number[];
    logPotential: (values: number[]) => number;
    neighbors: BayesVariable[];
}

export type BayesNodeInput = BayesVariableInput | BayesFactorInput;
export type BayesNode = BayesVariable | BayesFactor;

export type BayesGraph = InternalFactorGraph<BayesVariable, BayesFactor>

export class BeliefPropagation {
    public graph: BayesGraph;
    public messages: number[][][];
    private nextMessages: number[][][];
    private rawLogVariableMarginals: number[][];
    private dampingFactor: number;
    private evidence = new Map<BayesVariable, number>();
    
    constructor(graph: FactorGraph<BayesNodeInput>, options: { damping?: number } = {}) {
        this.graph = this.processGraph(graph);
        this.dampingFactor = options.damping || 0;
        this.messages = []
        this.nextMessages = []
        this.rawLogVariableMarginals = []
        this.evidence = new Map<BayesVariable, number>()
        this.initializeMessages();
    }

    private processGraph(graph: FactorGraph<BayesNodeInput>): BayesGraph {
        let nodes: [NodeID, BayesNode][] = Object.entries(graph.nodes).map(([nodeId, node], i) => (
            node.type === GraphType.VARIABLE ?
                [nodeId, Object.assign(
                    {},
                    node,
                    {logBelief: [], name: nodeId, id: i, neighbors: []}
                ) as BayesVariable]:
                [nodeId, Object.assign(
                    {},
                    node,
                    {name: nodeId, id: i, logPotential: (values: number[]) => {
                        const p = node.potential(values);
                        return p > 0 ? Math.log(p) : BeliefUtils.LOG_EPSILON;
                    }, neighbors: [], currentBelief: []}
                ) as BayesFactor]
        ))

        const nodeDict = Object.fromEntries(nodes)

        const edges = graph.edges
            .filter(
                ([from, to]) => from in nodeDict && to in nodeDict
            )
            .map(
                ([from, to]) => [nodeDict[from]?.id || 0, nodeDict[to]?.id || 0] as [number, number]
            )
            .flatMap(([from, to]) => [[from, to], [to, from]] as [number, number][])

        let nodesOut = nodes.map(([i, o]) => o);

        edges.forEach(([from, to]) => {
            nodesOut[from]?.type == GraphType.VARIABLE && nodesOut[to] && nodesOut[from]?.type != nodesOut[to]?.type ?
            nodesOut[from].neighbors.push(nodesOut[to]) :
            nodesOut[from]?.type == GraphType.FACTOR && nodesOut[to] && nodesOut[from]?.type != nodesOut[to]?.type ?
            nodesOut[from].neighbors.push(nodesOut[to]) :
                console.error(`Invalid edge: (${from}, ${to})`)
        })

        return { nodes: nodes.map(([_, o]) => o), edges };
    }

    private initializeMessages(): void {
        this.messages = Array(this.graph.nodes.length).fill([]).map(() => 
            Array(this.graph.nodes.length).fill([])
        );
        this.nextMessages = Array(this.graph.nodes.length).fill([]).map(() => 
            Array(this.graph.nodes.length).fill([])
        );
        this.rawLogVariableMarginals = Array(this.graph.nodes.length).fill([]);

        for (const node of this.graph.nodes) {
            if (node.type === GraphType.VARIABLE) {
                const size = node.possibleValues.length;
                const uniformProb = 1 / size;
                this.rawLogVariableMarginals[node.id] = Array(size).fill(Math.log(uniformProb));
                node.currentBelief = Array(size).fill(uniformProb);
            }
        }

        for (const [from, to] of this.graph.edges) {
            const toNode = this.graph.nodes[to];
            if (toNode?.type === GraphType.VARIABLE) {
                this.messages[from][to] = [...this.rawLogVariableMarginals[to]].map(Math.exp);
            } else {
                this.messages[from][to] = [1];
            }
        }
    }
    
    private updateVariableToFactorMessage(variable: BayesVariable, factor: BayesFactor): void {
        if (this.evidence.has(variable)) {
            this.nextMessages[variable.id][factor.id] = [...this.messages[variable.id][factor.id]];
            return;
        }

        const marginal = this.rawLogVariableMarginals[variable.id];
        const factorMessage = this.messages[factor.id][variable.id];
        
        // Divide out the target factor's contribution
        const message = marginal.map((m, i) => {
            const divisor = factorMessage[i] || BeliefUtils.LOG_EPSILON;
            return m - Math.log(divisor);
        }).map(Math.exp);
    
        this.nextMessages[variable.id][factor.id] = message;
    }

    private updateFactorToVariableMessage(factor: BayesFactor, variable: BayesVariable): void {
        const varIndex = factor.neighbors.indexOf(variable);
        const otherVars = factor.neighbors.filter(v => v !== variable);
    
        // Precompute log(mu_w→f) for all other variables w
        const logMessagesFromOtherVars = otherVars.map(w => 
            this.messages[w.id][factor.id].map(m => Math.log(Math.max(m, 1e-20)))
        );

        const valueCombinations = otherVars
            .map(w => w.possibleValues.map((_, i) => i))
            .reduce((acc, vals) => 
                acc.flatMap(combo => vals.map(v => [...combo, v])), 
                [[]] as number[][]
            );

        const logMessages = variable.possibleValues.map((_, valIndex) => (
            BeliefUtils.logSumExp(
                valueCombinations.map(combo => {
                    const values = new Array(factor.neighbors.length);
                    values[varIndex] = valIndex;
                    combo.forEach((v, i) => {
                        values[factor.neighbors.indexOf(otherVars[i])] = v;
                    });
    
                    return factor.logPotential(values) +
                        combo.reduce((sum, v, i) => sum + logMessagesFromOtherVars[i][v], 0);
                })
            )
        ));
    
        this.nextMessages[factor.id][variable.id] = logMessages.map(Math.exp);
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

        // Swap messages for next iteration
        [this.messages, this.nextMessages] = [this.nextMessages, this.messages];
    }


    private updateBeliefs(): void {
        for (const node of this.graph.nodes) {
            // Skip belief updates for observed nodes
            if (node.type == GraphType.VARIABLE && this.evidence.has(node)) continue;

            this.rawLogVariableMarginals[node.id] = node.currentBelief.map((_, i) => (
                node.neighbors
                    .map(neighbor => Math.log(this.messages[neighbor.id][node.id][i] || 1))
                    .reduce((a, b) => a + b, 0)
            ));

            node.currentBelief = BeliefUtils.normalize(BeliefUtils.dampMessage(
                this.rawLogVariableMarginals[node.id].map(Math.exp),
                node.currentBelief,
                this.dampingFactor
            ));
        }
    }
    
    public runIterations(maxIterations: number, tolerance: number = 1e-6): void {
        for (let i = 0; i < maxIterations; i++) {
            const oldBeliefs = this.graph.nodes
                .filter(node => node.type === GraphType.VARIABLE)
                .map(node => [...node.currentBelief]);
    
            this.updateAllMessages();
            this.updateBeliefs();
    
            const maxDiff = this.graph.nodes
                .filter(node => node.type === GraphType.VARIABLE)
                .reduce((maxDiff, node, index) => {
                    const diff = Math.max(...node.currentBelief.map((b, i) => Math.abs(b - oldBeliefs[index][i])));
                    return Math.max(maxDiff, diff);
                }, 0);
    
            if (maxDiff < tolerance) {
                console.log(`Converged after ${i + 1} iterations`);
                return;
            }
        }
        console.log(`Did not converge after ${maxIterations} iterations`);
    }

    /**
     * Sets evidence by clamping a variable to a specific value
     * @param nodeId - Name of the variable node
     * @param value - The observed value (must be in possibleValues)
     */
    public setEvidence(nodeId: string, value: number): void {
        const node = this.graph.nodes.find(n => n.name === nodeId);
        if (!node || node.type !== GraphType.VARIABLE) return;

        // Store evidence
        this.evidence.set(node, value);

        // Force beliefs to observed value
        node.currentBelief = node.possibleValues.map(v => v === value ? 1 : 0);
        this.rawLogVariableMarginals[node.id] = [...node.currentBelief].map(Math.log);

        // Initialize messages to reflect evidence
        for (const factor of node.neighbors) {
            this.messages[node.id][factor.id] = [...node.currentBelief];
            this.nextMessages[node.id][factor.id] = [...node.currentBelief];
        }
    }

    /**
     * Gets the current belief distribution for a node
     * @param nodeId - Name of the node
     * @returns Probability distribution over possible values
     */
    public getBeliefs(nodeId: string): number[] {
        const node = this.graph.nodes.find(n => n.name === nodeId);
        if (!node || node.type !== GraphType.VARIABLE) {
            throw new Error(`Cannot get beliefs from non-variable node: ${nodeId}`);
        }
        return [...node.currentBelief];
    }
}
