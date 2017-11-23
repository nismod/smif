import React, { Component } from 'react';
import PropTypes from 'prop-types'

class NarrativeSelector extends Component {
    constructor(props) {
        super(props)

        this.narrativeSelectHandler = this.narrativeSelectHandler.bind(this)
    }

    narrativeSelectHandler(event, narrative_name) {
        const {change_narrative} = this.props

        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name
        const key = narrative_name

        change_narrative(name, key, value)
    }

    render() {
        const {narrativeSet, narratives} = this.props

        return (
            <div>
                <fieldset>
                    <legend>{narrativeSet}</legend>
                    {
                        narratives.map((narrative) => (
                            <div key={narrative.name}>
                                <input type="checkbox" name={narrativeSet} key={narrative.name} defaultChecked={narrative.active} onClick={(event) => {this.narrativeSelectHandler(event, narrative.name);}}></input>
                                <label>{narrative.name}</label>
                            </div>
                        ))
                    }
                </fieldset>
            </div>

        )
    }
}

NarrativeSelector.propTypes = {
    narrativeSet: PropTypes.string.isRequired,
    narratives: PropTypes.array.isRequired,
    change_narrative: PropTypes.func.isRequired
};

export default NarrativeSelector;