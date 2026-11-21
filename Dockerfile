# Built by the scorer on linux/amd64. The digest is the image the commits were
# labeled in, so sqlglot's tests behave here as they did for the labels.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The exact package set the labels were made with.
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.lock

# sqlglot at the pinned commit, from its own repository.
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
ADD https://github.com/tobymao/sqlglot/archive/${SQLGLOT_SHA}.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /opt \
 && tar -xzf /tmp/sqlglot.tar.gz -C /opt \
 && mv /opt/sqlglot-${SQLGLOT_SHA} /opt/sqlglot \
 && rm /tmp/sqlglot.tar.gz \
 && python -m compileall -q -j 0 /opt/sqlglot/sqlglot /opt/sqlglot/tests \
 && chmod -R a+rX /opt/sqlglot

# Warm caches the run cannot write: pytest's rewritten test modules.
RUN cd /opt/sqlglot && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --collect-only -q \
      -p no:cacheprovider tests > /dev/null && chmod -R a+rX /opt/sqlglot

COPY app /app
RUN python /app/selector.py --build-index \
 && cd /opt/sqlglot && PYTHONPATH=/opt/sqlglot python /app/proxy.py --baseline /app/proxy_base.json \
 && PYTHONPATH=/opt/sqlglot python /app/impfp.py --baseline /app/impfp_base.json \
 && python -m compileall -q /app \
 && python -c "import py_compile; py_compile.compile('/app/selector.py', cfile='/app/main.pyc', doraise=True)" \
 && chmod -R a+rX /app

ENV SKIP_INTEGRATION=0 PYTHONUNBUFFERED=1
# A script given as .py is compiled on every start (0.05 s of CPU); its
# bytecode runs the same, with /app as sys.path[0] all the same.
CMD ["python", "/app/main.pyc"]
ENV BLAST_TPCDS_RISK=1
