#pragma once

#include "Vector.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <deque>

namespace LSTM {

	template <typename DataType>
	struct sigmoid {
		DataType operator()(DataType value) {
			return 1 / (1 + std::exp(-value));
		}
	};

	template <typename DataType>
	struct sigmoid_derive {
		DataType operator()(DataType value) {
			sigmoid<DataType> func;
			DataType ret = func(value);
			return (1 - ret) * ret;
		}
	};

	template <typename DataType>
	struct tanh {
		DataType operator()(DataType value) {
			return std::tanh(value);
		}
	};

	template <typename DataType>
	struct tanh_derive {
		DataType operator()(DataType value) {
			return 1 - std::pow(std::tanh(value), 2);
		}
	};

	template <std::size_t Input, std::size_t Output, std::size_t MaxMemory = 0, 
		typename DataType = float,
		typename ActivationFunc = tanh<DataType>,
		typename ActivationFuncGate = sigmoid<DataType>,
		typename ActivationFuncDerive = tanh_derive<DataType>,
		typename ActivationFuncGateDerive = sigmoid_derive<DataType>
	>
	class Node {
		typedef struct {
			Vector<Input, DataType> x;
			Vector<Output, DataType> z, i, f, c, o, y;
		}memory_t;
		typedef struct {
			Vector<Input, DataType> del_x;
			Vector<Output, DataType> del_z, del_i, del_f, del_c, del_o, del_y;
		}memory_back_t;

		std::deque<memory_t>	m_memory;

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

		Vector<Input, DataType> backprop(const memory_back_t& mem_t_1, std::size_t index) {

			ActivationFunc acfunc;
			ActivationFuncDerive acfunc_d;
			ActivationFuncGateDerive acfunc_g_d;

			auto data_t = m_memory.at(index);

			memory_back_t next_mem;

			auto del_y = mem_t_1.del_y
				+ Transpose(m_recWeight_input) * mem_t_1.del_z
				+ Transpose(m_recWeight_inputGate) * mem_t_1.del_i
				+ Transpose(m_recWeight_forgetGate) * mem_t_1.del_f
				+ Transpose(m_recWeight_outputGate) * mem_t_1.del_o;  

			next_mem.del_o = HadamardProduct(HadamardProduct(del_y, data_t.c.transform(acfunc)),
										data_t.o.transform(acfunc_g_d));

			next_mem.del_c = HadamardProduct(HadamardProduct(del_y, data_t.o), data_t.c.transform(acfunc_d))
				+ HadamardProduct(m_weight_peep_outputGate, next_mem.del_o)
				+ HadamardProduct(m_weight_peep_inputGate, mem_t_1.del_i)
				+ HadamardProduct(m_weight_peep_forgetGate, mem_t_1.del_f)
				+ HadamardProduct(mem_t_1.del_c, mem_t_1.del_f);

			next_mem.del_f = index > 0 ? 
				HadamardProduct(HadamardProduct(next_mem.del_c, m_memory.at(index - 1).c),
				data_t.f.transform(acfunc_g_d)) : 0;

			next_mem.del_i = HadamardProduct(HadamardProduct(next_mem.del_c, data_t.z),
				data_t.i.transform(acfunc_g_d));

			next_mem.del_z = HadamardProduct(HadamardProduct(next_mem.del_c, data_t.i),
				data_t.z.transform(acfunc_d));

			next_mem.del_x = Transpose(m_weight_input) * next_mem.del_z
				+ Transpose(m_weight_inputGate) * next_mem.del_i
				+ Transpose(m_weight_forgetGate) * next_mem.del_f
				+ Transpose(m_weight_outputGate) * next_mem.del_o;

			if (index > 0) {
				backprop(next_mem, index - 1);
			}

			m_weight_input += DirectProduct(next_mem.del_z, data_t.x);
			m_weight_inputGate += DirectProduct(next_mem.del_i, data_t.x);
			m_weight_forgetGate += DirectProduct(next_mem.del_f, data_t.x);
			m_weight_outputGate += DirectProduct(next_mem.del_o, data_t.x);

			m_recWeight_input += DirectProduct(mem_t_1.del_z, data_t.y);
			m_recWeight_inputGate += DirectProduct(mem_t_1.del_i, data_t.y);
			m_recWeight_forgetGate += DirectProduct(mem_t_1.del_f, data_t.y);
			m_recWeight_outputGate += DirectProduct(mem_t_1.del_o, data_t.y);

			m_bias_input += next_mem.del_z;
			m_bias_inputGate += next_mem.del_i;
			m_bias_forgetGate += next_mem.del_f;
			m_bias_outputGate += next_mem.del_o;

			m_weight_peep_inputGate += HadamardProduct(next_mem.del_c, mem_t_1.del_i);
			m_weight_peep_forgetGate += HadamardProduct(next_mem.del_c, mem_t_1.del_f);
			m_weight_peep_outputGate += HadamardProduct(next_mem.del_c, next_mem.del_o);

			return next_mem.del_x;
		}

	public:
		const Vector<Output, DataType>& Forward(const Vector<Input, DataType>& x) {
			auto y = !m_memory.empty() ? m_memory.back().y : 0;
			auto c = !m_memory.empty() ? m_memory.back().c : 0;

			memory_t mem;

			mem.z = m_weight_input * x + m_recWeight_input * y + m_bias_input;

			mem.i = m_weight_inputGate * x + m_recWeight_inputGate * y 
				+ HadamardProduct(m_weight_peep_inputGate, c) + m_bias_inputGate;

			mem.f = m_weight_forgetGate * x + m_recWeight_forgetGate * y 
				+ HadamardProduct(m_weight_peep_forgetGate, c) + m_bias_forgetGate;

			mem.c = HadamardProduct(mem.i.transform(ActivationFuncGate()), 
				mem.z.transform(ActivationFuncGate())) 
				+ HadamardProduct(mem.f.transform(ActivationFunc()), c);

			mem.o = m_weight_outputGate * x + m_recWeight_outputGate * y
				+ HadamardProduct(m_weight_peep_outputGate, mem.c) + m_bias_outputGate;

			mem.y = HadamardProduct(mem.o.transform(ActivationFuncGate()), 
				mem.c.transform(ActivationFunc()));

			m_memory.push_back(mem);
			while (MaxMemory > 0 && m_memory.size() > MaxMemory) {
				m_memory.pop_front();
			}

			return m_memory.back().y;
		}

		Vector<Input, DataType> Backward(const Vector<Output, DataType>& del) {
			memory_back_t mem_t_1;
			mem_t_1.del_y = del;
			return backprop(mem_t_1, m_memory.size() - 1);
		}
	};

}
