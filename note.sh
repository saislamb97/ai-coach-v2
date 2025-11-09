python3 manage.py makemigrations && python3 manage.py migrate && python3 manage.py collectstatic --noinput --clear && mkdir -p ../temp/logs && celery -A aicoach worker -l info --logfile=../temp/logs/celery_worker.log & celery -A aicoach beat -l info --logfile=../temp/logs/celery_beat.log & python3 manage.py runserver 0.0.0.0:8000

cd app && python3 manage.py makemigrations && python3 manage.py migrate && python3 manage.py collectstatic --noinput --clear

mkdir -p temp/logs && cd app && celery -A aicoach worker -l info --logfile=../temp/logs/celery_worker.log & celery -A aicoach beat -l info --logfile=../temp/logs/celery_beat.log & python3 manage.py runserver 0.0.0.0:8000
