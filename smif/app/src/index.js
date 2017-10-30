import React from 'react';
import { render } from 'react-dom';
import { Provider } from 'react-redux';
import { Route, HashRouter as Router } from 'react-router-dom';

import Nav from './components/Nav';
import Welcome from './components/Welcome';
import ProjectOverview from './components/ProjectOverview';
import SosModelRunConfig from './components/SosModelRunConfig';
import SosModelConfig from './components/SosModelConfig';

import { store } from './store';

import 'normalize.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../static/css/main.css';

render(
    <Provider store={store}>
        <Router>
            <div className="container">
                <Nav />
                <Route exact path="/" component={Welcome}/>
                <Route exact path="/configure" component={ProjectOverview}/>
                <Route path="/configure/sos-modelrun" component={SosModelRunConfig}/>
                <Route path="/configure/sos-model" component={SosModelConfig}/>
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
);
