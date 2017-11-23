import React, { Component } from 'react';
import PropTypes from 'prop-types'

class TimestepSelector extends Component {
    constructor(props) {
        super(props)

        this.timestepSelectHandler = this.timestepSelectHandler.bind(this)
        this.onChangeHandler = this.onChangeHandler.bind(this)

        this.state = {}
        this.state.timesteps = this.props.defaultValue
        this.state.baseyear = this.state.timesteps[0]
        this.state.endyear = this.state.timesteps[this.state.timesteps.length - 1]
        this.state.resolution = ((this.state.endyear - this.state.baseyear) / (this.state.timesteps.length - 1))
    }

    createBaseyearSelectItems() {
        const {timesteps} = this.state
        let years = []
            
        for (let i = 2000; i <= 2100; i++) {
            if (i == timesteps[0]) {
                years.push(<option key={'baseyear_' + i} selected="selected" value={i}>{i}</option>)
            }
            else {
                years.push(<option key={'baseyear_' + i} value={i}>{i}</option>)
            }
        }

        return years
    }

    createEndyearSelectItems() {
        const {timesteps} = this.state

        let years = []
        for (let i = this.state.baseyear; i <= this.state.baseyear + 100; i = i + this.state.resolution) {
            if (i == timesteps[timesteps.length - 1]) {
                years.push(<option key={'endyear_' + i} selected="selected" value={i}>{i}</option>)
            }
            else {
                years.push(<option key={'endyear_' + i} value={i}>{i}</option>)
            }
        }
        return years;
    }

    timestepSelectHandler(event) {
        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        this.state[name] = parseInt(value)
        this.onChangeHandler()
    }

    onChangeHandler() {
        const {onChange} = this.props
        const {baseyear, endyear, resolution} = this.state

        let timesteps = []

        for (let i = baseyear; i <= endyear; i+=resolution)
        {
            timesteps.push(i)
        }
        onChange(timesteps)
    }

    render() {
        const {defaultValue} = this.props

        return (
            <div>
                <label>Base year:</label>
                <div className="select-container">
                    <select name="baseyear" onChange={this.timestepSelectHandler}>
                        <option value="" disabled="disabled">Please select a base year</option>
                        {this.createBaseyearSelectItems()}
                    </select>
                </div>
                <label>End year:</label>
                <div className="select-container">
                    <select name="endyear" onChange={this.timestepSelectHandler}>
                        <option value="" disabled="disabled">Please select an end year</option>
                        {this.createEndyearSelectItems()}
                    </select>
                </div>
                <label>Resolution:</label>
                <input name="resolution" type="number" min="1" defaultValue={this.state.resolution} onChange={this.timestepSelectHandler}/>
            </div>
        )
    }
}

TimestepSelector.propTypes = {
    defaultValue: PropTypes.array.isRequired,
    onChange: PropTypes.func.isRequired
};

export default TimestepSelector;