import React from 'react';
import { render } from 'react-dom';
import { Provider } from 'react-redux';
import { Route, HashRouter as Router} from 'react-router-dom';

import App from './components/App';
import ProjectOverview from './components/ProjectOverview';
import SosModelRunConfig from './components/SosModelRunConfig';
import SosModelConfig from './components/SosModelConfig';

import { store } from './store';

import 'normalize.css';
import '../static/css/main.css';

render(
    <Provider store={store}>
        <Router>
            <div>
                <Route exact path="/" component={App}/>
                <Route exact path="/configure" component={ProjectOverview}/>
                <Route path="/configure/sos-modelrun" component={SosModelRunConfig}/>
                <Route path="/configure/sos-model" component={SosModelConfig}/>
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
);
