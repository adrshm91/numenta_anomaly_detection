from clearml.automation import PipelineController

pipe = PipelineController(
    name="IForest pipeline",
    project="numenta_anomaly_detection",
    version="0.0.1",
    add_pipeline_tags=False,
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="stage_preprocess",
    base_task_project="numenta_anomaly_detection",
    base_task_name="preprocess",
)

pipe.add_step(
    name="stage_train",
    base_task_project="numenta_anomaly_detection",
    base_task_name="training",
    parents=["stage_preprocess"],
)

# pipe.start_locally(run_pipeline_steps_locally=True)

pipe.start()

print("done")
