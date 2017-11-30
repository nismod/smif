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
                for (let k = 0; i < activeProperties.length; k++) {
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
        const {onChange} = this.props

        onChange(target.name, target.checked)
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

    renderWarning(message) {
        return (
            <div className="alert alert-danger">
                {message}
            </div>
        )
    }

    render() {
        const {name, activeProperties, availableProperties} = this.props

        if (availableProperties == null || availableProperties.length == 0) {
            return this.renderWarning('There are no ' + name + ' properties available')
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