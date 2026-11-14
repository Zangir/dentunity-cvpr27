"""Read members of a remote zip via HTTP range requests (no full download)."""
import io, os, ssl, urllib.request, zipfile
import certifi

CTX = ssl.create_default_context(cafile=certifi.where())


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=CTX) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, context=CTX) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 20))
