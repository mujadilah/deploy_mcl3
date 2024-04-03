import os
 
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, 
    StatisticsGen, 
    SchemaGen, 
    ExampleValidator, 
    Transform, 
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2 
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)


 
# return components

def init_components(
    data_dir,
    transform_module,
    training_module,
    serving_model_dir,
):
    """Initiate tfx pipeline components
 
    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory
 
    Returns:
        TFX components
    """
    output_hasil = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )
 
    example_gen = CsvExampleGen(
        input_base=data_dir, 
        output_config=output_hasil
    )

    gen_statik = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )
    
    skema_gen = SchemaGen(
        statistics=gen_statik.outputs["statistics"]
    )
    contoh_validator = ExampleValidator(
        statistics=gen_statik.outputs['statistics'],
        schema=skema_gen.outputs['schema']
    )

    
    komponen_transform  = Transform(
        examples=example_gen.outputs['examples'],
        schema= skema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )
    
    komponen_trainer  = Trainer(
        module_file=os.path.abspath(training_module),
        examples = komponen_transform.outputs['transformed_examples'],
        transform_graph=komponen_transform.outputs['transform_graph'],
        schema=skema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train']),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval']),
        custom_config={'batch_size': 32}

    )
    
    model_resolver = Resolver(
        strategy_class= LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Liked')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[

                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value':0.5}
                        ),change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value':0.0001}
                        )
                    )
                )
            ])
        ]

    )
        
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=komponen_trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=komponen_trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='serving_model_dir'
            )
        )
    )
    
    components = (
        example_gen,
        gen_statik,
        skema_gen,
        contoh_validator,
        komponen_transform,
        komponen_trainer,
        model_resolver,
        evaluator,
        pusher
    )
    
    return components