import React, { Component } from 'react'
import PropTypes from 'prop-types'

class PropertySelector extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
    }

    flagActiveSectorModels(activeProperties, availableProperties) {

        for (let i = 0; i < availableProperties.length; i++) {
            availableProperties[i].active = false

            if (activeProperties != null) {
                for (let k = 0; k < activeProperties.length; k++) {
                    if (activeProperties[k] == availableProperties[i].name) {
                        availableProperties[i].active = true
                        break
                    }
                }
            }
        }

        return availableProperties
    }

    handleChange(event) {
        const target = event.target
        const {name, onChange} = this.props

        let newProperties = this.props.activeProperties

        // initialize properties if not already done
        if (newProperties == null && newProperties == undefined) {
            newProperties = []
        }

        // add or remove the property
        if (target.checked) {
            newProperties.push(target.name)
        } else {
            for (let i = 0; i < newProperties.length; i++) {
                if (newProperties[i]== target.name) {
                    newProperties.splice(i, 1)
                    break
                }
            }
        }

        // create an event structure
        let returnEvent = {
            target: {
                name: name,
                value: newProperties,
                type: 'array'
            }
        }

        // send event
        onChange(returnEvent)
    }

    renderProperySelector(selectedProperties) {
        return (
            <div>
                <div className="card">
                    <div className="card-body">

                        {
                            Object.keys(selectedProperties).map((i) => (
                                <div className="form-check" key={i}>
                                    <label className="form-check-label">
                                        <input className="form-check-input" type="checkbox" name={selectedProperties[i].name} key={selectedProperties[i].name} value={selectedProperties[i].name} defaultChecked={selectedProperties[i].active} onClick={this.handleChange}></input>
                                        {selectedProperties[i].name}
                                    </label>
                                </div>
                            ))
                        }
                    </div>
                </div>
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="property_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="property_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="property_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {name, activeProperties, availableProperties} = this.props

        if (availableProperties == null || availableProperties.length == 0) {
            return this.renderInfo('There are no ' + name + ' properties available')
        } else {
            let selectedProperties = this.flagActiveSectorModels(activeProperties, availableProperties)
            return this.renderProperySelector(selectedProperties)
        }
    }
}

PropertySelector.propTypes = {
    name: PropTypes.string,
    activeProperties: PropTypes.array,
    availableProperties: PropTypes.array,
    onChange: PropTypes.func
}

export default PropertySelector
