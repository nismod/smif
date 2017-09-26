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

## Getting started links

- [React](https://facebook.github.io/react/docs/hello-world.html) view layer
- [Redux](http://redux.js.org/) data flow library
- [D3](https://github.com/d3/d3/wiki) visualisation library
- [Webpack](https://webpack.js.org/) build tool

## Development notes

Recommend installing [eslint](https://eslint.org/) (`npm install --global eslint`),
which is configured on a project level by `.eslintrc.json`. There is also a [VS Code
extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint).

