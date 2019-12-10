const dotProduct = (t1, t2) => t1.reduce((total, item, index) => total + item * t2[index] , 0);
const sigmoid = (val) => 1 / (1 + Math.exp(-val))
const genRandVec = (length, multiplier = 2, bias = 1) => Array(length).fill(() => (Math.random() * multiplier) - bias).map(x => x());
const geneticMerge = (num1, num2) => {
  const base = Math.random()
  return num1*base + num2*(1-base);
}
const NN = ({
  inputs,
  hiddenLayers,
  outputs,
  weights,
  biases,
  activationFunction = sigmoid
}) => {
  if (!weights) {
    weights = [];
     
     layers = [inputs, ...hiddenLayers];
    layers.forEach((layer, idx) => {
      nextLayer = layers[idx + 1];
      if (nextLayer === undefined) {
        nextLayer = outputs;
      }
      weights.push(Array(nextLayer).fill(() => genRandVec(layer)).map(x => x().concat(1)))
    });
    // console.log("Weights: ", weights)
  }
  if(!biases) {
    biases = genRandVec(hiddenLayers.length + 1, 5, 2.5)
    // console.log("Biases : ", biases)
  }
  return {
    feedForward(inputs){
      weights.forEach((weightList, idx) => {
        inputs = inputs.concat(biases[idx]);
        const outputs = [];
        weightList.forEach((weights) => {
          outputs.push(sigmoid(dotProduct(weights, inputs)))
        })
        inputs = outputs;
        // console.log(inputs)
      })
      return inputs;
    },
    weights() {
      return weights;
    },
    biases() {
      return biases;
    },
    reproduce(change) {
      return NN({
        weights: weights.map(weightList => weightList.map(weights => weights.map(weight => weight === 1 ? 1 : weight + Math.random() * change - change / 2))),
        biases: biases.map(bias => bias + (Math.random() * change * 3) - ((change * 3) / 2))
      })
    },
    reproduceWith(nn) {
      const otherWeights = nn.weights();
      const otherBiases = nn.biases();
      return NN({
        weights: weights.map((weightList, i1) => {
          return weightList.map((weightRow, i2) => {
            return weightRow.map((weight, i3) => geneticMerge(weight, otherWeights[i1][i2][i3]))
          })
        }),
        biases: biases.map((bias, idx) => geneticMerge(bias, otherBiases[idx]))
      })
    }
  }
}
