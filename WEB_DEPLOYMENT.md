# Browser-native deployment

The NCA playground has no application backend. The UI calls a browser-side compatibility service in `static/js/modules/browserNcaService.js`, so the application is a static site.

## Local development

Because the JavaScript uses ES modules, serve the repository through any static HTTP server rather than opening `index.html` with a `file://` URL. For example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

Python is only being used as a convenient static file server in that example; the application itself executes entirely in the browser.

## GitHub Pages

A Pages workflow is included in `.github/workflows/pages.yml`. It publishes only `index.html` and `static/`, keeping the Pages artifact focused on the files required by the running application.

After applying the patches and pushing `main`:

1. Open **Settings → Pages** in the GitHub repository.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Push to `main` (or run **Deploy GitHub Pages** manually from the Actions tab).

The workflow then deploys the static playground under the repository's GitHub Pages URL.
