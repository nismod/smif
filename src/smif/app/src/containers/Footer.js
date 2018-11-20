import React, { Component } from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'

import { fetchSmifDetails } from 'actions/actions.js'

class Footer extends Component {

    constructor(props) {
        super(props)
        this.init = true
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
            <div className="container-fluid">
                <div className="row">
                    <div className="col pb-3">
                        smif version: <span className="badge badge-dark">{version}</span>
                    </div>
                </div>
            </div>
        )
    }

    render() {
        const {isFetching, smif} = this.props

        if (isFetching && this.init) {
            return this.renderLoading()
        } else {
            this.init = false
            return this.renderFooter(smif.version)
        }
    }
}

Footer.propTypes = {
    smif: PropTypes.object.isRequired,
    isFetching: PropTypes.bool.isRequired,
    dispatch: PropTypes.func.isRequired
}

function mapStateToProps(state) {
    const { smif } = state

    return {
        smif: smif.item,
        isFetching: (smif.isFetching)
    }
}

export default connect(mapStateToProps)(Footer)
