import glob

from tensorflow.python.eager import context

from models.predefined_simulator import PredefinedSimulator
from autodist.strategy import base
from autodist.resource_spec import ResourceSpec

from pathlib import Path

GRAPH_ITEM_DIR = f'/tmp/autodist/original-graph'
SIMULATION_DATA_DIR = f'{str(Path.home())}/autosync_dataset_release/vgg16'
CHECKPOINT_DIR =  f'{str(Path.home())}'

#TODO: change to vgg
resource_spec_file = f'{SIMULATION_DATA_DIR}/cluster1/vgg16_aws_4_pure_random/resource_spec.yml'
original_graph_item_path = '/tmp/autodist/original-graph'
checkpoint_path = '/home/jongho/vgg16_predefined_checkpoints/ckpV1_vgg_aws_100_0.78375_0.76375'
strategy_dir = f'{SIMULATION_DATA_DIR}/cluster1/vgg16_aws_4_pure_random/strategies'
strategy_files = glob.glob(f'{strategy_dir}/*')
strategy_file = strategy_files[0]


with context.graph_mode():

    strategy = base.Strategy.deserialize(strategy_file)

    simulator = PredefinedSimulator(original_graph_item_path=original_graph_item_path)

    cost = simulator.simulate(strategy=strategy, resource_spec=ResourceSpec(resource_spec_file), checkpoint=checkpoint_path)

    print(f"strategy_file: {strategy_file}, cost: {cost}")


print('finished')
