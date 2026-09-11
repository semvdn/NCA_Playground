# Browser-native deployment

The NCA playground has no application backend. The UI calls a browser-side compatibility service in `static/js/modules/browserNcaService.js`, so the application is a static site.

## Local development

Because the JavaScript uses ES modules, serve the repository through any static HTTP server rather than opening `index.html` with a `file://` URL. For example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

Python is only a convenient static file server in that example; the application itself executes entirely in the browser.

## Tests

The project has no npm runtime dependencies. Node is used only for the regression suite:

```bash
npm test
```

## GitHub Pages

`.github/workflows/pages.yml` runs the engine tests on pushes and pull requests. On `main`, deployment only proceeds after those tests pass. The deployment artifact contains only `index.html` and `static/`.

In **Settings → Pages**, set **Build and deployment → Source** to **GitHub Actions**. Pushing `main` then tests and deploys the static playground automatically.
