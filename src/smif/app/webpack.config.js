const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const MiniCssExtractPlugin = require('mini-css-extract-plugin')
const CleanWebpackPlugin = require('clean-webpack-plugin')

module.exports = {
    entry: {
        // Define entry points for each major part of the application.
        //
        // Initially, 'main' is for the application, 'patternLibrary' for the
        // design-only pattern library.
        // The keys of this object are used as the chunk names ('main' and
        // 'patternLibrary')
        main: './src/index.js',
        patternLibrary: './src/pattern-library.js'
    },
    devServer: {
        contentBase: '.',
        // serve 'index.html' for other routes, so that live reload works even
        // with other routes
        historyApiFallback: true,
        // proxy API requests to another local server (assuming the python
        // flask app is running to serve API requests)
        proxy: {
            '/api/*': {
                target: 'http://localhost:5000'
            }
        }
    },
    output: {
        // Output javascript files to the 'dist' directory
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist'),
        publicPath:'/'
    },
    module: {
        rules: [
            // Use MiniCssExtractPlugin to pull in CSS.
            // NB: there are pre- and post-processors available for CSS, not
            // used here.
            {
                test: /\.css$/,
                use: [
                    {
                        loader: MiniCssExtractPlugin.loader
                    },
                    'css-loader'
                ]
            },

            // Pre-process javascript from ES6 to target current browsers.
            // - 'react' means that JSX syntax is processed
            // - 'env' uses settings in `.babelrc` to target particular browsers
            //   (see https://babeljs.io/docs/plugins/preset-env for details)
            {
                test: /\.jsx?$/,
                loader: 'babel-loader',
                exclude: /node_modules/,
                query: {
                    presets: ['@babel/preset-react', '@babel/preset-env']
                }
            }
        ]
    },
    resolve: {
        alias: {
            actions: path.resolve('src', 'actions'),
            components: path.resolve('src', 'components'),
            containers: path.resolve('src', 'containers'),
            reducers: path.resolve('src', 'reducers'),
            store: path.resolve('src', 'store'),
        }
    },
    plugins: [
        // Output an `index.html` file with the 'main' application chunk
        new HtmlWebpackPlugin({
            title: 'smif',
            filename: './index.html',
            hash: true,
            template: 'src/index.html',
            chunks: ['main']
        }),

        // Output a 'pattern-library.html` file with the 'patternLibrary' chunk
        new HtmlWebpackPlugin({
            title: 'smif - Pattern library',
            filename: './pattern-library.html',
            hash: true,
            template: 'src/index.html',
            chunks: ['patternLibrary']
        }),

        // Register plugin to process CSS
        new MiniCssExtractPlugin(),

        // Clean the `dist` directory on each run to ensure all files are
        // generated and old generated files are cleaned out.
        new CleanWebpackPlugin(['dist'])
    ],
};
