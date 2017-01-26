#include "LSTM.hpp"
#include <iostream>

const float data[][3] = {
	{0, 0, 0},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 1},
};

int main() {
	LSTM::Node<2,1> n1;

	float err_sum = 0;
	do {
		err_sum = 0;
		for (int i = 0; i < _countof(data); i++) {
			auto ret = n1.Forward({ data[i][0], data[i][1] });
			auto err = data[i][2] - ret.at(0);
			err_sum += err * err;
			n1.Backward({ err });
		}
		std::cout << err_sum << std::endl;
	} while (err_sum > 0.01f);

	return 0;
}
