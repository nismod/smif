import React from 'react'
import { render } from 'react-dom'
import { Provider } from 'react-redux'
import { Route, Router, browserHistory} from 'react-router';

import App from './components/App'
import ProjectList from './components/ProjectList'
import ProjectOverview from './components/ProjectOverview'
import SosModelRunConfig from './components/SosModelRunConfig'
import SosModelConfig from './components/SosModelConfig'
import SimulationModel from './components/SimulationModelConfig'

import { store } from './store';

render(
  <Provider store={store}>
    <Router history={browserHistory}>
      <Route path="/" component={App}/>
      <Route path="/configure" component={ProjectOverview}/>
      <Route path="/configure/sos-modelrun" component={SosModelRunConfig}/>
      <Route path="/configure/sos-model" component={SosModelConfig}/>
    </Router>
  </Provider>,
  document.getElementById('root')
)