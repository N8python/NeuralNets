const dotProduct = (t1, t2) => t1.reduce((total, item, index) => total + item * t2[index] , 0);
const sigmoid = (val) => 1 / (1 + Math.exp(-val))
const genRandVec = (length, multiplier = 2, bias = 1) => Array(length).fill(() => (Math.random() * multiplier) - bias).map(x => x());
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
    }
  }
}
