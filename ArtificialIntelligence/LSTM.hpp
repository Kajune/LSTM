#pragma once

#include "Vector.hpp"
#include "Matrix.hpp"
#include <cmath>

namespace LSTM {

	template <typename DataType>
	struct sigmoid {
		DataType operator()(DataType value) {
			return 1 / (1 + std::exp(-value));
		}
	};

	template <typename DataType>
	struct tanh {
		DataType operator()(DataType value) {
			return std::tanh(value);
		}
	};

	template <std::size_t Input, std::size_t Output, 
		typename DataType = float,
		typename ActivationFunc = tanh<DataType>,
		typename ActivationFuncGate = sigmoid<DataType>>
	class Node {
		Vector<Output, DataType>	m_innerState, m_lastOutput;

		Matrix<Output, Input, DataType>
			m_weight_input,
			m_weight_inputGate,
			m_weight_forgetGate,
			m_weight_outputGate;

		Matrix<Output, Output, DataType>
			m_recWeight_input,
			m_recWeight_inputGate,
			m_recWeight_forgetGate,
			m_recWeight_outputGate;

		Vector<Output, DataType>
			m_weight_peep_inputGate,
			m_weight_peep_forgetGate,
			m_weight_peep_outputGate;

		Vector<Output, DataType>
			m_bias_input,
			m_bias_inputGate,
			m_bias_forgetGate,
			m_bias_outputGate;

	public:
		const Vector<Output, DataType>& Forward(const Vector<Input, DataType>& input) {
			auto z = m_weight_input * input + m_recWeight_input * m_lastOutput + m_bias_input;

			auto i = m_weight_inputGate * input + m_recWeight_inputGate * m_lastOutput 
				+ DirectProduct(m_weight_peep_inputGate, m_innerState) + m_bias_inputGate;

			auto f = m_weight_forgetGate * input + m_recWeight_forgetGate * m_lastOutput 
				+ DirectProduct(m_weight_peep_forgetGate, m_innerState) + m_bias_forgetGate;

			m_innerState = DirectProduct(i.transform(ActivationFuncGate()), 
				z.transform(ActivationFuncGate())) 
				+ DirectProduct(f.transform(ActivationFunc()), m_innerState);

			auto o = m_weight_outputGate * input + m_recWeight_outputGate * m_lastOutput
				+ DirectProduct(m_weight_peep_outputGate, m_innerState) + m_bias_outputGate;

			m_lastOutput = DirectProduct(o.transform(ActivationFuncGate()), 
				m_innerState.transform(ActivationFunc()));

			return m_lastOutput;
		}
	};

}
