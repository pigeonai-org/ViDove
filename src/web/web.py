import yaml
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from src.task import Task
from uuid import uuid4

app = Flask(__name__)

# Global thread pool
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as per your requirement

# thread safe task map to store task status
task_map = {}

@app.route('/api/task', methods=['POST'])
def create_task_youtube():
    global task_map
    data = request.get_json()
    if not data or 'youtubeLink' not in data:
        return jsonify({'error': 'YouTube link not provided'}), 400
    youtube_link = data['youtubeLink']
    launch_config = yaml.load(open("./configs/local_launch.yaml"), Loader=yaml.Loader)
    task_id = str(uuid4())
    task = Task.fromYoutubeLink(youtube_link, task_id, launch_config)
    task_map[task_id] = task
    # Submit task to thread pool
    executor.submit(task.run)

    return jsonify({'taskId': task.task_id})

@app.route('/api/task/<taskId>/status', methods=['GET'])
def get_task_status(taskId):
    global task_map
    if taskId not in task_map:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({'status': task_map[taskId].status})

if __name__ == '__main__':
    app.run(debug=True)
