python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition tpr &
python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition fnr &
python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition par &
python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition fpr &
python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition tnr &
python -Wi main.py --model RISAN --cost 0.125 --dataset Adult --gamma 1 --fairness_condition nar &