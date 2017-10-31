# User Interface

This directory contains the front-end of the graphical user interface, which
communicates with the `smif` core through the `smif.http-api`.

## Dependencies

- [node.js](https://nodejs.org/en/download/) for development and building.

To install javascaript library dependencies and build the client:

```bash
cd path/to/smif/app/
npm install    # downloads packages from https://docs.npmjs.com/ to node_modules
npm run build  # runs build script specified in package.json
npm start      # runs build, watches for changes, and opens in browser
```

To develop against the python HTTP API, run the API server against test fixtures
in another terminal with:

```bash
cd ../http_api/
FLASK_APP=smif.http_api.app FLASK_DEBUG=1 flask run
```

Run tests with:

```bash
npm run test
```

## Getting started links

- [React](https://facebook.github.io/react/docs/hello-world.html) view layer
- [Redux](http://redux.js.org/) data flow library
- [D3](https://github.com/d3/d3/wiki) visualisation library
- [Webpack](https://webpack.js.org/) build tool

## Development notes

Recommend installing [eslint](https://eslint.org/) (`npm install --global eslint`),
which is configured on a project level by `.eslintrc.json`. There is also a [VS Code
extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint).

### Code structure

We're using [React](https://facebook.github.io/react/docs/hello-world.html)
components, which live in `./src/components`, to provide the low-level building
block of the user interface. Components are 'pure' in that they render their
data, and provide and respond to events without knowledge of their context in
the rest of the application.

Components are combined in containers in `./src/containers`, which group
sections of the interface by related functionality, and are aware of the data
layer provided by Redux, which will involve 'actions' (which signal events) and
'reducers' (which respond by changing the app state).

See the [Redux docs](http://redux.js.org/docs/introduction/Examples.html)
for examples of increasing complexity.

### Pattern library

The elements and components of the application should be included in the
pattern library, for ease of reference while designing the look and feel of the
app.

When running a local development server (`npm start`), visit
http://localhost:8080/pattern-library.html or after building the application,
(`npm run build`), open `./dist/pattern-library.html`.

### Testing

Tests are run using the [mocha](https://mochajs.org/) test framework, with
[chai](http://chaijs.com/) for assertions and [enzyme](http://airbnb.io/enzyme/)
to wrap React components.
