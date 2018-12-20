# User Interface

This directory contains the front-end of the graphical user interface, which
communicates with the `smif` core through the `smif.http-api`.

## Getting started

### Setup

1. Install [smif](https://github.com/nismod/smif) as described in the [installation and configuration documentation](https://smif.readthedocs.io/en/latest/installation.html#installation-and-configuration).

2. Install [node.js](https://nodejs.org/en/download/) for development and building.

3. Install javascript library dependencies:

```bash
cd src/smif/app/
npm install    # downloads packages from https://docs.npmjs.com/ to node_modules
```

### Run the smif app

1. Build the client:

```bash
npm run build  # runs build script specified in package.json
```

2. Run the smif app to serve the python HTTP API:

```bash
smif app -d src/smif/sample_project/
```

3. Browse to http://localhost:5000 where the app is now hosted.

## Run smif app in debug mode

1. Start `smif app` with debug flag and smif verbosity for more detailed debug information:

```bash
FLASK_DEBUG=1 smif -v app -d src/smif/sample_project
```

2. Run the front-end in npm to watch code changes and automatically build and restart the app during runtime:

```bash
npm start      # runs build, watches for changes, and opens in browser
```

3. The browser will now automatically open http://localhost:8080 where `smif app` is hosted.

## Development notes

### Environment

[NodeJS](https://nodejs.org/) is used to execute the JavaScript code. This run-time environment executes code on the client instead of in the browser and helps to manage packages using the pre-installed npm package manager.

[Webpack](https://webpack.js.org/) is used to bundle Javascript files for usage in a browser.

### Development stack

*Smif app* is developed in [JavaScript](https://developer.mozilla.org/bm/docs/Web/JavaScript) using the the [React](https://reactjs.org/) library to build reusable user-interface components and [Redux](https://redux.js.org/) to manage the application state. [Bootstrap](https://getbootstrap.com/) HTML and CSS based design templates are used to design the front-end. Numerous react optimised components were used to enhance rapid development, such as [react-select](https://www.npmjs.com/package/react-select) and [react-table](https://www.npmjs.com/package/react-table).

A full overview of the used packages can be found in the `smif/src/smif/app/package.json` file.

### Code style

Recommend installing [eslint](https://eslint.org/) (`npm install --global eslint`),
which is configured on a project level by `.eslintrc.json`. There is also a [VS Code
extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint).

### Documentation

The smif app is documented within the smif documentation. The GUI screenshots can be generated from a linux computer running [cutycapt](http://cutycapt.sourceforge.net/) and [ImageMagick](https://imagemagick.org/), there is a script in `docs/gui/screenshot.sh` that automates taking screenshots and putting labels on them.

### Browser plugins

There are browser plugins available such as the [React Developer Tools](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/) for Firefox that allows to inspect a React tree, including the component hierarchy, props, state and more.

### Testing

Tests are run using the [mocha](https://mochajs.org/) test framework, with
[chai](http://chaijs.com/) for assertions and [enzyme](http://airbnb.io/enzyme/)
to wrap React components. The tests are in `smif/src/smif/app/tests` and can be run using the `npm test` command.

## Project overview

The *smif app* project has the following layout:
- components
- containers
- actions
- reducers
- store

### Components

We're using [React](https://facebook.github.io/react/docs/hello-world.html)
components, which live in `./src/components`, to provide the low-level building
block of the user interface. Components are 'pure' in that they render their
data, and provide and respond to events without knowledge of their context in
the rest of the application.

### Containers

Components are combined in containers in `./src/containers`, which group
sections of the interface by related functionality, and are aware of the data
layer provided by Redux, which will involve 'actions' (which signal events) and
'reducers' (which respond by changing the app state).

See the [Redux docs](http://redux.js.org/docs/introduction/Examples.html)
for examples of increasing complexity.

### Actions
Actions are payloads of information that send data from your application to your store. They are the only source of information for the store. You send them to the store using store.dispatch(). These are specified in `./src/actions/actions.js`.

### Reducers
Reducers specify how the application's state changes in response to actions sent to the store. The reducers are implemented in `./src/reducers/reducers.js`.

### Store
A store holds the whole state tree of the application and is implemented in `./src/store/store.js`.

## Connection to Smif

The *smif app* communicates with smif through a web API. The default address of the API is *http://localhost:5000/api/v1/*, an overview of configurations can be obtained (GET) by extending this url with the configuration type such as *model_runs*, *sector_models* or *scenarios* f.e. *http://localhost:5000/api/v1/model_runs/*. Crud operations can be performed on individual configurations by extending the url with the configuration name such as *http://localhost:5000/api/v1/model_runs/energy_central*.

The information that is exchanged is expressed in JSON. A full overview of the API can be found in `smif/src/smif/http_api/crud.py`.
