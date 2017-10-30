import React from 'react'
import { render } from 'react-dom'
import { Provider } from 'react-redux'
import { Route, BrowserRouter as Router} from 'react-router-dom';

import App from './components/App'
import ProjectOverview from './components/ProjectOverview'
import SosModelRunConfig from './components/SosModelRunConfig'
import SosModelConfig from './components/SosModelConfig'
import SimulationModel from './components/SimulationModelConfig'

import { store } from './store';

render(
  <Provider store={store}>
    <Router>
      <div>
      <Route path="/" component={App}/>
      <Route path="/configure" component={ProjectOverview}/>
      <Route path="/configure/sos-modelrun" component={SosModelRunConfig}/>
      <Route path="/configure/sos-model" component={SosModelConfig}/>
      </div>
    </Router>
  </Provider>,
  document.getElementById('root')
)
