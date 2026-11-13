import json
try:
    import simplejson as json
except ImportError:
    import json
import requests
import os.path
import stat
import uuid


def autodetect_proxy():
    proxies = {}

    proxy_https = os.getenv('HTTPS_PROXY', os.getenv('https_proxy', None))
    proxy_http = os.getenv('HTTP_PROXY', os.getenv('http_proxy', None))

    if proxy_https:
        proxies['https'] = proxy_https
    if proxy_http:
        proxies['http'] = proxy_http

    return proxies


def fetch_dict(endpoint):
    data = None
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = autodetect_proxy()

        if proxies:
            response = requests.get(url=endpoint, proxies=proxies)
        else:
            response = requests.get(url=endpoint)

        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            data = {}
    else:
        data = {}
        if os.path.exists(endpoint):
            with open(endpoint) as fd:
                try:
                    data = json.load(fd)
                except json.JSONDecodeError:
                    data = {}

    return data


def save_dict(endpoint, data):
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = autodetect_proxy()

        if proxies:
            _ = requests.patch(url=endpoint, headers={'content-type':'application/json'}, data=json.dumps(data, indent=4, sort_keys=True), proxies=proxies)
        else:
            _ = requests.patch(url=endpoint, headers={'content-type':'application/json'}, data=json.dumps(data, indent=4, sort_keys=True))
    else:
        # Never truncate a previously saved result: a watchdog/SIGKILL during
        # serialization must leave either the old or the new complete file.
        tmp_path = str(endpoint) + '.' + uuid.uuid4().hex + '.tmp'
        try:
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            with os.fdopen(fd, 'w') as stream:
                if os.path.exists(endpoint):
                    os.fchmod(stream.fileno(), stat.S_IMODE(os.stat(endpoint).st_mode))
                json.dump(data, stream, indent=4)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, endpoint)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
