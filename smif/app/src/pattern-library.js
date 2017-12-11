import React from 'react';
import { render } from 'react-dom';

import PatternLibrary from './containers/PatternLibrary/pattern-library.js';

import 'bootstrap/dist/css/bootstrap.min.css';
import '../static/css/main.css';

render(
    <PatternLibrary />,
    document.getElementById('root')
);
