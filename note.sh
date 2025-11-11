cd app && python3 manage.py makemigrations && python3 manage.py migrate && python3 manage.py collectstatic --noinput --clear
cd app && celery -A aicoach worker -l info & cd app && celery -A aicoach beat -l info & cd app && python3 manage.py runserver 0.0.0.0:8000

python3 manage.py runserver 0.0.0.0:8000 & celery -A aicoach worker -l info & celery -A aicoach beat -l info & wait