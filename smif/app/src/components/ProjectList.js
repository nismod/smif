import React, { Component } from 'react'
import { connect } from 'react-redux'

import { fetchProjects, fetchProjectsSuccess } from '../actions'

import '../../static/css/main.css';

export class ProjectList extends Component {
    constructor(props) {
        super(props);
        props.configurations.projects = ["loading"]
        this.somevariable = ["hallo", "hello"];
    }

    componentWillMount(){
        this.props.fetchProjects();
    }
    
    render() {
        return (
            <div className="content-wrapper">
                <div hidden>
                    <h3>Loading...</h3>
                </div>
        
                <div hidden>
                    <h3>Error</h3>
                </div>
        
                <div>
                    <input type="button" value="Start a new project" />
                    <label>Projectname:</label>
                    <input type="text" list="models" name="myModels" />
                    <datalist id="models">

                        <option value="Project 1"></option>
                        <option value="Project 2"></option>
                        <option value="Project 3"></option>
                        <option value="Project 4"></option>
                    </datalist>
                    <input type="button" value="Go" />

                    <h2>{this.props.configurations.projects[0]}</h2>
                    <h2>{this.props.configurations.projects[1]}</h2>
                    
                </div>
            </div>
        )
    }
}

// ProjectListContainer.js
function mapStateToProps(state, ownProps) {
    return {
        configurations: state.configurations
    };
}

const mapDispatchToProps = {
    fetchProjects
};

const ProjectListContainer = connect(mapStateToProps, mapDispatchToProps)(ProjectList);

export default ProjectListContainer;