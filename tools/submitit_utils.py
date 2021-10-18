import submitit


def launch_job(func, args):
    executor = submitit.AutoExecutor(folder=args.work_dir)

    executor.update_parameters(
        # mem_gb=int(16 * args.num_gpus / args.num_machines),
        gpus_per_node=int(args.num_gpus / args.num_machines),
        # tasks_per_node=int(args.num_gpus / args.num_machines),
        # tasks_per_node=1,
        # cpus_per_task=args.workers,
        # nodes=args.num_machines,
        timeout_min=3 * 24 * 60,
        slurm_partition=args.partition,
        slurm_constraint=args.gpu_type,
        name=args.name,
    )

    job = executor.submit(func, args)
    print(f"Job id: j{job.job_id}")
