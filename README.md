# Neural Cellular Automata Playground

An interactive, browser-native playground for exploring neural cellular automata (NCA). Each cell updates from its wrapped 3x3 neighborhood using a small configurable MLP, making it easy to see how local neural rules produce large-scale spatial dynamics.

**Live app:** https://semvdn.github.io/NCA_Playground/

<img width="1042" height="807" alt="NCA_README_img" src="https://github.com/user-attachments/assets/2d745bc0-a92d-4daa-b143-78b8993ef31f" />


## What is a neural cellular automaton?

A cellular automaton updates a grid by applying the same local rule at every cell. In this playground, that local rule is neural rather than handwritten: a Multi-Layer Perceptron receives the 9 values in a cell's 3x3 neighborhood and produces the cell's next scalar state.

The grid wraps at the boundaries, so every cell always has nine inputs. Hidden layers can use ReLU, sigmoid, or tanh activations, while the final output is passed through a sigmoid to keep the state in `[0, 1]`.

## Visual showcase

Some earlier simulations and UI captures:

https://github.com/user-attachments/assets/193f7d69-e515-4e92-ac71-ed1806af617c

https://github.com/user-attachments/assets/9955a7e6-54ed-43b2-b862-1cfa6ab4e89e

https://github.com/user-attachments/assets/9593cd5c-74c5-4153-ba6e-c907327c4107

https://github.com/user-attachments/assets/3921b359-9f6f-4fb2-a14e-8a36d1ff1c52

Web UI walkthrough: https://youtu.be/euN4uQ0BBNc

## License

See [`LICENSE`](LICENSE).

## AI Use

Generative AI, primarily OpenAI's GPT 5.3 codex, was used extensively as a development tool throughout this project.
