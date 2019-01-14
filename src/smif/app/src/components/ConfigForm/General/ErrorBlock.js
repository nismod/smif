import React from 'react'
import PropTypes from 'prop-types'

const ErrorBlock = (props) => {
    if (props.errors){
        return (
            <div className="alert alert-danger">
                <p>{ (props.intro)? props.intro : 'Errors:' }</p>
                <ul>
                    {
                        props.errors.map((exception, idx) => (
                            <li key={idx}>
                                {`${exception.error} ${exception.message}`}
                            </li>
                        ))
                    }
                </ul>
            </div>
        )
    } else {
        return null
    }
}

ErrorBlock.propTypes = {
    errors: PropTypes.array,
    intro: PropTypes.string
}

export default ErrorBlock
