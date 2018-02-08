import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { Link, Router } from 'react-router-dom'

import { fetchSmifDetails } from '../actions/actions.js'

class Footer extends Component {
    
    constructor(props) {
        super(props)

    }

    componentDidMount() {
        const { dispatch } = this.props
        dispatch(fetchSmifDetails())
    }

    renderLoading() {
        return (
            <div className="alert alert-primary">
                Loading...
            </div>
        )
    }

    renderError() {
        return (
            <div className="alert alert-danger">
                Error
            </div>
        )
    }

    renderFooter(version) {
        return (  
            <div className="container">
                <div className="row justify-content-md-center">
                    <div className="col-md-auto">
                        <span className="badge badge-dark">{version}</span>
                    </div>
                </div>
            </div>
        )
    }

    render() {
        const {isFetching, smif} = this.props

        if (isFetching) {
            return this.renderLoading()
        } else {
            return this.renderFooter(smif.version)
        }
    }
}

Footer.propTypes = {
    smif: PropTypes.object.isRequired
}

function mapStateToProps(state) {
    const { smif } = state

    return {
        smif: state.smif.item,
        isFetching: (state.smif.isFetching)
    }
}

export default connect(mapStateToProps)(Footer)
