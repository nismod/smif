const path = require('path');
const webpack = require('webpack')
const HtmlWebpackPlugin = require('html-webpack-plugin');
const ExtractTextPlugin = require('extract-text-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');

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
        contentBase: '.'
    },
    output: {
        // Output javascript files to the 'dist' directory
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist')
    },
    module: {
        rules: [
            // Use ExtractTextPlugin to pull in CSS.
            // NB: there are pre- and post-processors available for CSS, not
            // used here.
            {
                test: /\.css$/,
                use: ExtractTextPlugin.extract({
                    fallback: 'style-loader',
                    use: 'css-loader'
                })
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
                    presets: ['react', 'env']
                }
            }
        ]
    },
    plugins: [
        // Extract all common code to 'common' js/css files
        new webpack.optimize.CommonsChunkPlugin({
            name: 'common',
            filename: 'common.js',
            chunks: ['main', 'patternLibrary']
        }),

        // Output an `index.html` file with the 'main' application chunk
        new HtmlWebpackPlugin({
            title: 'smif',
            filename: './index.html',
            hash: true,
            template: 'src/index.ejs',
            chunks: ['common', 'main']
        }),

        // Output a 'pattern-library.html` file with the 'patternLibrary' chunk
        new HtmlWebpackPlugin({
            title: 'smif - Pattern library',
            filename: './pattern-library.html',
            hash: true,
            chunks: ['common', 'patternLibrary'],
            template: 'src/index.html'
        }),

        // Register ExtractTextPlugin to process CSS
        new ExtractTextPlugin('[name].css'),

        // Clean the `dist` directory on each run to ensure all files are
        // generated and old generated files are cleaned out.
        new CleanWebpackPlugin(['dist'])
    ]
};
