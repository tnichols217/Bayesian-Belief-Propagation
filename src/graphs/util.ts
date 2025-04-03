export type NodeID = string;

export enum GraphType {
    FACTOR,
    VARIABLE
}

export interface GraphNode {
    type: GraphType;
    description?: string;
}

export interface GraphNodeInternal {
    name: NodeID;
    id: number;
}

export interface FactorNode extends GraphNode {
    type: GraphType.FACTOR;
}

export interface VariableNode extends GraphNode {
    type: GraphType.VARIABLE;
}

export interface FactorGraph<T extends GraphNode, Q extends NodeID = NodeID> {
    nodes: Record<NodeID, T>;
    edges: [Q, Q][];
}

export type Edge<R, Q> = [R, Q] | [Q, R];

export interface InternalFactorGraph<
    R extends VariableNode,
    Q extends FactorNode,
    Nodes extends (R|Q)[] = (R|Q)[],
    Edges extends [number, number][] = [number, number][]
> {
    nodes: Nodes;
    edges: Edges;
}

export type GaussianMessage = { mean: number; precision: number };

export const BeliefUtils = {
    LOG_EPSILON: 1e-10,
    
    logSumExp(logValues: number[]): number {
        if (logValues.length === 0) return this.LOG_EPSILON;
        const maxLogValue = Math.max(...logValues);
        if (maxLogValue === -Infinity) return this.LOG_EPSILON;
        let sum = 0;
        for (const logVal of logValues) {
            sum += Math.exp(logVal - maxLogValue);
        }
        return maxLogValue + Math.log(sum);
    },

    logNormalize(logValues: number[]): number[] {
        const logSum = this.logSumExp(logValues);
        return logValues.map(lv => lv - logSum);
    },

    dampMessage(
        newMsg: number[],
        prevMsg: number[],
        dampingFactor: number
    ): number[] {
        const damped = newMsg.map((val, idx) => 
            dampingFactor * val + (1 - dampingFactor) * prevMsg[idx]
        );
        const sum = damped.reduce((a, b) => a + b, 0);
        return damped.map(x => x / sum);
    },

    gaussianDampMessage(
        newMsg: GaussianMessage,
        prevMsg: GaussianMessage,
        dampingFactor: number
    ): GaussianMessage {
        return {
            mean: dampingFactor * newMsg.mean + (1 - dampingFactor) * prevMsg.mean,
            precision: dampingFactor * newMsg.precision + (1 - dampingFactor) * prevMsg.precision
        }
    },

    normalize(values: number[]): number[] {
        const sum = values.reduce((a, b) => a + b, 0);
        return values.map(x => x / sum);
    },

    cartesianProduct<T>(arrays: T[][]): T[][] {
        return arrays.reduce<T[][]>(
            (acc, arr) => acc.flatMap(c => arr.map(v => [...c, v])),
            [[]]
        );
    }
};