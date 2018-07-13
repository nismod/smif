import React, { Component } from 'react'
import PropTypes from 'prop-types'

/**
 * IntroBlock
 *
 * Styled heading block used at the top of major pages
 */
class IntroBlock extends Component {
    render() {
        const {
            title,
            intro
        } = this.props

        return <div className="jumbotron jumbotron-fluid">
            <h1>{title}</h1>
            <p className="lead">{intro}</p>
            {this.props.children}
        </div>
    }
}

IntroBlock.propTypes = {
    title: PropTypes.string.isRequired,
    intro: PropTypes.string.isRequired,
    children: PropTypes.element.isRequired
}

export default IntroBlock
