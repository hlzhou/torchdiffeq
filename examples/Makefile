run_point_process:
	python point_processes.py --dataset poisson            --niters 5000 --jump_type read --batch_size 30 --nsave 500 --seed0;
	python point_processes.py --dataset exponential_hawkes --niters 5000 --jump_type read --batch_size 30 --nsave 500 --seed0;
	python point_processes.py --dataset self_inhibiting    --niters 5000 --jump_type read --batch_size 30 --nsave 500 --seed0;
	python point_processes.py --dataset powerlaw_hawkes    --niters 5000 --jump_type read --batch_size 30 --nsave 500 --seed0;

run_point_process_with_real_valued_label:
	python point_processes_rvl.py --datase exponential_hawkes --niter 5000 --jump_type read --batch_size 30 --nsave 500 --seed0 &

run_stack_overflow:
	python stack_overflow.py --niters 1500 --fold 0 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python stack_overflow.py --niters 1500 --fold 1 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python stack_overflow.py --niters 1500 --fold 2 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python stack_overflow.py --niters 1500 --fold 3 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python stack_overflow.py --niters 1500 --fold 4 --jump_type read --batch_size 30 --nsave 500 --seed0 &

run_mimic2:
	python mimic2.py --niters 3000 --fold 0 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python mimic2.py --niters 3000 --fold 1 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python mimic2.py --niters 3000 --fold 2 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python mimic2.py --niters 3000 --fold 3 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	python mimic2.py --niters 3000 --fold 4 --jump_type read --batch_size 30 --nsave 500 --seed0 &
