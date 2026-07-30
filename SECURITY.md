# Security Policy

## Supported versions

mlango is pre-1.0. Security fixes land on the latest release.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Email **drobishev.denis@icloud.com** with:

- what the issue is and what an attacker could do with it
- the smallest reproduction you can manage
- the mlango and Python versions involved

You can expect an acknowledgement within 72 hours and an assessment within
seven days. If a fix is warranted, we will agree a disclosure timeline with you
and credit you in the release notes unless you prefer otherwise.

## Things to know before deploying

mlango's defaults are tuned for a laptop, not the public internet. The
following are deliberate development conveniences that you must change:

**The admin is unauthenticated by default.** Set `ADMIN_PASSWORD` to require
HTTP Basic auth, or — better — put the admin behind your existing identity
provider. `manage.py check` warns when the admin is open and `DEBUG` is off.

**The inference API is unauthenticated by default.** Add
`mlango.serve.middleware.ApiKeyMiddleware` to `SERVE_MIDDLEWARE` and populate
`SERVE_API_KEYS`, or terminate auth at your gateway.

**`SECRET_KEY` is generated per project and belongs in your secret store**, not
in version control.

**`DEBUG = True` shows tracebacks.** Turn it off outside development.

**`RateLimitMiddleware` is per-process.** It stops a runaway script; it is not a
substitute for a gateway.

**Model artifacts are executable code.** `Model.load()` unpickles a joblib file
or loads a torch checkpoint. Only load artifacts your own runs produced —
loading an untrusted checkpoint is equivalent to running untrusted code.

**Data migrations run arbitrary Python.** `RunPython` operations execute on
`manage.py migrate`, so review migrations from outside contributors as you would
review any other code.

**Tools an agent can call run with your process's privileges.** A tool that
shells out, writes files, or calls an internal API gives the model that reach.
Validate tool inputs and gate anything destructive.

## Known non-issues

- The `echo` provider is deterministic and offline by design; it is a test
  double, not a model.
- Admin pages render values from your own datasets. If your data contains
  untrusted HTML, note that Jinja2 autoescaping is on — values are escaped, not
  rendered as markup.
