import { BeliefPropagation, type BayesNodeInput } from "./graphs/bayes";
import { GraphType, type FactorGraph } from "./graphs/util";

const nodes = {
    // Disease node (hidden cause)
    "disease": {
        type: GraphType.VARIABLE,
        possibleValues: [0, 1, 2], // 0=Healthy, 1=Cold, 2=Flu
        currentBelief: [0.8, 0.1, 0.1] // Prior: 80% healthy
    } as BayesNodeInput,
    
    // Symptom nodes (observations)
    "fever": {
        type: GraphType.VARIABLE,
        possibleValues: [0, 1], // 0=No, 1=Yes
        currentBelief: [0.5, 0.5] // Uniform initial
    } as BayesNodeInput,
    "cough": {
        type: GraphType.VARIABLE,
        possibleValues: [0, 1], // 0=No, 1=Yes
        currentBelief: [0.5, 0.5]
    } as BayesNodeInput,
    
    // Factors encoding medical knowledge
    "disease_to_fever": {
        type: GraphType.FACTOR,
        potential: ([d, f]) => {
            // P(fever|disease)
            const probs = [
                [0.99, 0.01], // Healthy
                [0.7, 0.3],    // Cold
                [0.1, 0.9]     // Flu
            ];
            return probs[d][f];
        }
    } as BayesNodeInput,
    "disease_to_cough": {
        type: GraphType.FACTOR,
        potential: ([d, c]) => {
            // P(cough|disease)
            const probs = [
                [0.9, 0.1],   // Healthy
                [0.2, 0.8],    // Cold
                [0.3, 0.7]     // Flu
            ];
            return probs[d][c];
        }
    } as BayesNodeInput
}

// Medical Diagnosis Example (Cold/Flu Network)
const diagnosisGraph: FactorGraph<BayesNodeInput, keyof typeof nodes> = {
    nodes,
    edges: [
        ["disease", "disease_to_fever"],
        ["fever", "disease_to_fever"],
        ["disease", "disease_to_cough"],
        ["cough", "disease_to_cough"]
    ]
};

// Create BP instance with damping for stability
const bp = new BeliefPropagation(diagnosisGraph, { damping: 0.3 });

// Observe symptoms
bp.setEvidence("fever", 1); // Patient has fever
bp.setEvidence("cough", 1); // Patient has cough

// Run inference
bp.runIterations(200);

// Get disease probabilities
const diseaseBelief = bp.getBeliefs("disease");
console.log("Diagnosis Probabilities:");
console.log(`Healthy: ${(diseaseBelief[0] * 100).toFixed(1)}%`);
console.log(`Cold:    ${(diseaseBelief[1] * 100).toFixed(1)}%`);
console.log(`Flu:     ${(diseaseBelief[2] * 100).toFixed(1)}%`);

const loopyGraph: FactorGraph<BayesNodeInput> = {
    nodes: {
        "A": {
            type: GraphType.VARIABLE,
            possibleValues: [0, 1],
            currentBelief: [0.5, 0.5]
        },
        "B": {
            type: GraphType.VARIABLE,
            possibleValues: [0, 1],
            currentBelief: [0.5, 0.5]
        },
        "C": {
            type: GraphType.VARIABLE,
            possibleValues: [0, 1, 2],
            currentBelief: [0.5, 0.3, 0.2]
        },
        "D": {
            type: GraphType.VARIABLE,
            possibleValues: [0, 1, 2],
            currentBelief: [0.5, 0.3, 0.2]
        },
        "AB": {
            type: GraphType.FACTOR,
            potential: ([a, b]) => a === b ? 0.99 : 0.01
        },
        "BC": {
            type: GraphType.FACTOR,
            potential: ([b, c]) => b === c ? 0.99 : 0.01
        },
        "CD": {
            type: GraphType.FACTOR,
            potential: ([c, d]) => c === d ? 0.7 : 0.3
        },
        "AD": {
            type: GraphType.FACTOR,
            potential: ([a, d]) => a === d ? 0.7 : 0.3
        }
    },
    edges: [
        ["A", "AB"], ["B", "AB"],
        ["B", "BC"], ["C", "BC"],
        ["C", "CD"], ["D", "CD"],
        ["A", "AD"], ["D", "AD"]
    ]
};

// Create BP instance with higher damping for loopy graphs
const bp2 = new BeliefPropagation(loopyGraph, { damping: 0.3 });

// Observe one node's state
bp2.setEvidence("A", 1);

bp2.runIterations(1000);
console.log("Node A:", bp2.getBeliefs("A"));
console.log("Node B:", bp2.getBeliefs("B"));
console.log("Node C:", bp2.getBeliefs("C"));
console.log("Node D:", bp2.getBeliefs("D"));
