import os
from flask import *
from werkzeug import secure_filename
from celery import Celery
from vec2pca import vec2pca
import glob
import pandas as pd


RESULTS_FOLDER = 'results'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# redis_conn = Redis()
# q = Queue(connection=redis_conn)


def results_files():
    return glob.glob(os.path.join(app.config['RESULTS_FOLDER'], '*.*'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def load_table(filename, rows=25):
    with open(os.path.join(app.config['RESULTS_FOLDER'], filename)) as f:
        table = f.read().replace('class="dataframe"', 'class="centered striped"')
        splitup = table.split('</thead>')
        index = splitup[0] + '</thead>'
        body = splitup[1].replace('<tbody>', '').split('</tbody>')[0]
        head = index + "</tr>".join(body.split("</tr>")[:rows]) + "</tr>" + '</tbody>'
        tail = index + "<tr>" + "<tr>".join(body.split("</tr>")[-rows:]) + '</tbody>'
        # import pdb; pdb.set_trace()
        return table, head, tail


@celery.task(bind=True)
def long_task(self, upload, result):
    

    self.update_state(state='PROGRESS',
                          meta={'current': 0, 'total': 0,
                                'status': "Started"})

    _df, df = vec2pca(upload, result)

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    

@app.route('/longtask', methods=['POST'])
def longtask():
    task = long_task.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    fnames = [os.path.split(n)[1] for n in results_files()]
    if request.method == 'POST':

        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)
            upload_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            results_file = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if "." in results_file[-6:]:
                results_file = '.'.join(results_file.split('.')[:-1]) + '.html'
            else:
                results_file = results_file + '.html'

        if upload_file != 'uploads/':
            async_result = long_task.delay(upload_file, results_file[:-5])
            df = async_result.get()
            # job   inputdata = q.enqueue(vec2pca, upload_file, results_file[:-5])
            components = []
            for i in range(1,9):
                components.append({
                    "name": "PC%d" % i,
                    "top": " ".join(df.iloc[0:200,i]),
                    "bottom": " ".join(list(df.iloc[-200:,i])[::-1])})

            #ajax_table, thead, ttail = load_table(os.path.split(results_file)[1])
            return render_template('upload.html', components=components)
    return render_template('upload.html', filenames="", thead="", ttail="")

@app.route('/ajax', methods=['POST'])
def dropdown():
    filename = request.form['filename']
    table, thead, ttail = load_table(filename)
    return jsonify(filename=table, thead=thead, ttail=ttail)

@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results/<filename>')
def result(filename):
    df = pd.read_csv(os.path.join(app.config['RESULTS_FOLDER'], filename))

    components = []
    for i in range(1,9):
        components.append({
        "name": "PC%d" % i,
        "top": "  ".join(df.iloc[0:200,i]),
        "bottom": "  ".join(list(df.iloc[-200:,i])[::-1])})
    print(components[0])
    return render_template('result.html', components=components)

if __name__ == "__main__":
    app.run(debug=True)

