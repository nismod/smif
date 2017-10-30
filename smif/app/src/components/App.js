import React, { Component } from 'react';
import { connect } from 'react-redux';

import { fetchProjects } from '../actions';

import Welcome from './Welcome'
import ProjectListContainer from './ProjectList'

export class App extends Component {

  	render() {
    	return (
      		<div>
				<Welcome />
				<ProjectListContainer />
	      	</div>
    	);
  	}
}

// AppContainer.js
const mapStateToProps = (state, ownProps) => ({
});

const mapDispatchToProps = {
};

const AppContainer = connect(
  mapStateToProps,
  mapDispatchToProps
)(App);

export default AppContainer;
