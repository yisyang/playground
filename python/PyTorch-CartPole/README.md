# PyTorch Bullets Env


## Installation

- Clone repo
- Install requirements
  - `pip install -r requirements.txt`
- While inside of project directory, install bullets env
  - `pip install -e gym-envs`


## Execution

  - For training
    - `python bullets-sb3.py -m train -ts 10000`
  - For training with multiple envs
    - `python bullets-sb3.py -m train -n 4 -ts 10000`
  - For playing with trained model
    - `python bullets-sb3.py -m ai`
  - For playing with manual input (arrow keys + z/x OR wasd + j/k)
    - `python bullets-sb3.py -m human -ds 1`


### Controls

(For manual human play)

  - Keyboard only
    - Movement: arrow keys or `w`/`a`/`s`/`d`
    - Charge weapon: hold and release `z` or `j`
    - Charge shield: hold and release `x` or `k`


## License
  MIT