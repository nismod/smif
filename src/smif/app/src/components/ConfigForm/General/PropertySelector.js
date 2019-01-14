import React, { Component } from 'react'
import PropTypes from 'prop-types'

class PropertySelector extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
    }

    flagActiveSectorModels(activeProperties, availableProperties) {

        // compose list of unique active and available properties
        let available_prop_names = availableProperties.map(availableProperties => availableProperties.name)
        let unique_properties = activeProperties.concat(available_prop_names).filter((v, i, a) => (a.indexOf(v) === i))

        // compose object with active and valid flags
        let flagged_props = []
        for (var propname of unique_properties.sort()) {
            flagged_props.push({
                name: propname,
                active: activeProperties.includes(propname) ? true : false,
                valid: available_prop_names.includes(propname) ? true : false
            })
        }

        return flagged_props
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
                <ul className="list-group">
                    <li className="list-group-item">
                        {
                            Object.keys(selectedProperties).map((i) => (
                                <div className="form-check" key={i}>
                                    <input
                                        className="form-check-input"
                                        type="checkbox"
                                        name={selectedProperties[i].name}
                                        key={selectedProperties[i].name}
                                        value={selectedProperties[i].name}
                                        defaultChecked={selectedProperties[i].active}
                                        onClick={this.handleChange}>
                                    </input>
                                    {
                                        selectedProperties[i].valid ?
                                            selectedProperties[i].name
                                            :
                                            <strike>
                                                {selectedProperties[i].name}
                                            </strike>
                                    }

                                </div>
                            ))
                        }
                    </li>
                </ul>
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

        if (availableProperties == null || availableProperties.length + activeProperties.length == 0) {
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
