#include <gtest/gtest.h>

#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/aten_xla_type_instances.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

class AtenXlaTensorTest : public TorchXlaTest {
 protected:
  static void SetUpTestCase() {
    AtenXlaType::RegisterAtenTypes();
    XlaHelpers::set_mat_mul_precision(xla::PrecisionConfig::HIGHEST);
  }
};

at::Tensor GetTestTensor(at::IntList sizes) {
  return at::rand(sizes, at::TensorOptions(at::kFloat));
}

void TestBackward(
    const std::vector<at::Tensor>& inputs, const Device& device,
    const std::function<at::Tensor(const std::vector<at::Tensor>&)>& testfn,
    double rtol = 1e-5, double atol = 1e-8,
    const std::vector<bool>& inputs_require_grad = {}) {
  CHECK(inputs_require_grad.empty() ||
        inputs.size() == inputs_require_grad.size());
  std::vector<at::Tensor> input_vars;
  std::vector<at::Tensor> xinput_vars;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& input = inputs[i];
    if (input.defined()) {
      const bool requires_grad =
          inputs_require_grad.empty() ? true : inputs_require_grad[i];
      input_vars.push_back(
          torch::autograd::make_variable(input, requires_grad));

      at::Tensor xinput = bridge::CreateXlaTensor(CopyTensor(input), device);
      xinput_vars.push_back(
          torch::autograd::make_variable(xinput, requires_grad));
    } else {
      input_vars.emplace_back();
      xinput_vars.emplace_back();
    }
  }

  at::Tensor output = testfn(input_vars);
  at::Tensor xoutput = testfn(xinput_vars);
  output.backward();
  xoutput.backward();
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (inputs[i].defined() && input_vars[i].requires_grad()) {
      ASSERT_TRUE(xinput_vars[i].grad().defined());
      AllClose(input_vars[i].grad(), xinput_vars[i].grad(), rtol, atol);
    }
  }
}

TEST_F(AtenXlaTensorTest, TestCastByte) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Byte(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Byte(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastShort) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Short(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Short(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastInt) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Int(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Int(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastLong) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Long(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Long(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestCastFloat) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::_cast_Float(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::_cast_Float(xla_a);
    EXPECT_TRUE(EqualValues(b, xla_b));
  });
}

TEST_F(AtenXlaTensorTest, TestRetainType) {
  at::Tensor xla_a =
      at::zeros({2, 2}, at::TensorOptions(at::kByte).device(at::kXLA));
  at::Tensor xla_b =
      at::ones({2, 2}, at::TensorOptions(at::kByte).device(at::kXLA));
  at::Tensor xla_c = xla_a + xla_b;
  EXPECT_EQ(xla_c.scalar_type(), at::ScalarType::Byte);
}

TEST_F(AtenXlaTensorTest, TestAdd) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::add(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAddInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.add_(b);
    at::Tensor xla_c = xla_a.add_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMul) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::mul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::mul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMulInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.mul_(b);
    at::Tensor xla_c = xla_a.mul_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDiv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::div(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::div(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestDivInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_a = bridge::CreateXlaTensor(a.clone(), device);
    at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor c = a.div_(b);
    at::Tensor xla_c = xla_a.div_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestRsub) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor other = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar alpha(2.5);
  at::Tensor result = at::rsub(input, other, alpha);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_other = bridge::CreateXlaTensor(other, device);
    at::Tensor xla_result = at::rsub(xla_input, xla_other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRsubScalar) {
  at::Tensor input = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Scalar other(1.5);
  at::Scalar alpha(2.5);
  at::Tensor result = at::rsub(input, other, alpha);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::rsub(xla_input, other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestNe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = GetTestTensor({2, 3});
  at::Tensor c = at::ne(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ne(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEq) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::eq(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::eq(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::ge(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::ge(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLe) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = a.clone();
  at::Tensor c = at::le(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::le(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestGt) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::gt(b, a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::gt(xla_b, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestLt) {
  at::Tensor a = GetTestTensor({2, 3});
  at::Tensor b = at::add(a.clone(), at::ones_like(a));
  at::Tensor c = at::lt(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::lt(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestNeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(0));
  at::Tensor result = at::ne(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::ne(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEqScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::eq(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::eq(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestGeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::ge(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::ge(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLeScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1));
  at::Tensor result = at::le(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::le(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestGtScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(0.5));
  at::Tensor result = at::gt(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::gt(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestLtScalar) {
  at::Tensor input = at::ones({2, 3});
  at::Scalar other(float(1.5));
  at::Tensor result = at::lt(input, other);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::lt(xla_input, other);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestIntegerAdd) {
  std::vector<at::ScalarType> types(
      {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong});

  ForEachDevice([&](const Device& device) {
    for (auto type : types) {
      at::Tensor a = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor b = at::randint(0, 63, {2, 2}, at::TensorOptions(type));
      at::Tensor c = at::add(b, 1.0);

      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = at::add(xla_b, 1.0);

      EXPECT_TRUE(EqualValues(c, ToCpuTensor(xla_c)));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestKthValue) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = at::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::kthvalue(xla_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestTopK) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool largest : {false, true}) {
        auto b = at::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::topk(xla_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSort) {
  at::Tensor a = at::rand({4, 5, 3}, at::TensorOptions(at::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = at::sort(a, dim, descending);
        ForEachDevice([&](const Device& device) {
          at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
          auto xla_b = at::sort(xla_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::min(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::min(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::max(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::max(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestAll) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::all(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::all(xla_a);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAllDim) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::all(a, 1, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::all(xla_a, 1, /*keepdim=*/false);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAllDimKeep) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::all(a, 1, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::all(xla_a, 1, /*keepdim=*/true);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAny) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::any(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::any(xla_a);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAnyDim) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::any(a, 1, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::any(xla_a, 1, /*keepdim=*/false);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAnyDimKeep) {
  at::Tensor a = at::randint(0, 5, {2, 3, 4}, at::TensorOptions(at::kByte));
  at::Tensor b = at::any(a, 1, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::any(xla_a, 1, /*keepdim=*/true);
    EqualValues(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMean) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::mean(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::mean(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMeanInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::mean(a, {1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::mean(xla_a, {1});
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestMeanInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::mean(a, {0, 1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::mean(xla_a, {0, 1});
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSum) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sum(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sum(a, {1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a, {1});
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumInDims) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sum(a, {0, 1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a, {0, 1});
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSumInDimsKeep) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sum(a, {0, 1}, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sum(xla_a, {0, 1}, /*keepdim=*/true);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProd) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::prod(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::prod(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProdInDim) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::prod(a, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::prod(xla_a, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestProdInDimKeep) {
  at::Tensor a = at::rand({4, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::prod(a, 1, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::prod(xla_a, 1, /*keepdim=*/true);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMin) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMinDim) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a, 1, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a, 1, /*keepdim=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMinDimKeep) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a, 1, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a, 1, /*keepdim=*/true);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMinSameValue) {
  at::Tensor a = at::ones({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmin(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMax) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMaxDim) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a, 1, /*keepdim=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a, 1, /*keepdim=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMaxDimKeep) {
  at::Tensor a = at::rand({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a, 1, /*keepdim=*/true);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a, 1, /*keepdim=*/true);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestArgMaxSameValue) {
  at::Tensor a = at::ones({4, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::argmax(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::argmax(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAsin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::asin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::asin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sin(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSinh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sinh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAcos) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::acos(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::acos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestCos) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::cos(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::cos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestCosh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::cosh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::cosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestAtan) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::atan(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::atan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestTan) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::tan(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::tan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestTanh) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::tanh(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::tanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMinMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, 0.311, 0.409);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, 0.311, 0.409);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMin) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, 0.311, c10::nullopt);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, 0.311, c10::nullopt);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestClampMax) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::clamp(a, c10::nullopt, 0.409);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::clamp(xla_a, c10::nullopt, 0.409);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestCeil) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::ceil(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::ceil(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestFloor) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::floor(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::floor(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestNeg) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::neg(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::neg(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestSign) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::sign(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sign(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestAbs) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::abs(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::abs(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestZeros) {
  at::Tensor a = at::zeros({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor xla_a =
      at::zeros({2, 2}, at::TensorOptions(at::kFloat).device(at::kXLA));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestOnes) {
  at::Tensor a = at::ones({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor xla_a =
      at::ones({2, 2}, at::TensorOptions(at::kFloat).device(at::kXLA));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestFull) {
  at::Tensor a = at::full({2, 2}, 3.1165, at::TensorOptions(at::kFloat));
  at::Tensor xla_a =
      at::full({2, 2}, 3.1165, at::TensorOptions(at::kFloat).device(at::kXLA));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestARange) {
  at::Tensor a = at::arange(0.0, 100.0, 0.5, at::TensorOptions(at::kFloat));
  at::Tensor xla_a = at::arange(0.0, 100.0, 0.5,
                                at::TensorOptions(at::kFloat).device(at::kXLA));
  AllClose(a, xla_a);
}

TEST_F(AtenXlaTensorTest, TestSigmoid) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::sigmoid(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x1) {
  at::Tensor a = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x1) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_1x2) {
  at::Tensor a = at::rand({4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmul_2x2) {
  at::Tensor a = at::rand({2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestMatmulBcast) {
  at::Tensor a = at::rand({4, 2, 3, 2, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 1, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::matmul(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestBatchMatMul) {
  at::Tensor a = at::rand({3, 6, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 4, 5}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::bmm(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::bmm(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumOuter) {
  at::Tensor a = at::rand({5}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({5}, at::TensorOptions(at::kFloat));
  std::string equation = "i,j->ij";
  at::Tensor c = at::einsum(equation, {a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumBatchMatMul) {
  at::Tensor a = at::rand({3, 2, 5}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 5, 4}, at::TensorOptions(at::kFloat));
  std::string equation = "bij,bjk->bik";
  at::Tensor c = at::einsum(equation, {a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBilinear) {
  at::Tensor a = at::rand({3, 5, 4}, at::TensorOptions(at::kFloat));
  at::Tensor l = at::rand({2, 5}, at::TensorOptions(at::kFloat));
  at::Tensor r = at::rand({2, 4}, at::TensorOptions(at::kFloat));
  std::string equation = "bn,anm,bm->ba";
  at::Tensor c = at::einsum(equation, {l, a, r});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_l = bridge::CreateXlaTensor(l, device);
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_r = bridge::CreateXlaTensor(r, device);
    at::Tensor xla_c = at::einsum(equation, {xla_l, xla_a, xla_r});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerDiagonal) {
  at::Tensor input = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  std::string equation = "ii->i";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchDiagonal) {
  at::Tensor input = at::rand({4, 3, 3}, at::TensorOptions(at::kFloat));
  std::string equation = "...ii->...i";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerBatchPermute) {
  at::Tensor input = at::rand({2, 3, 4, 5}, at::TensorOptions(at::kFloat));
  std::string equation = "...ij->...ji";
  at::Tensor result = at::einsum(equation, {input});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_result = at::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestEinsumPyTorchLowerRepeatedAxis) {
  at::Tensor x = at::rand({2, 3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor y = at::rand({4}, at::TensorOptions(at::kFloat));
  std::string equation = "ijj,k->ik";
  at::Tensor result = at::einsum(equation, {x, y});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_x = bridge::CreateXlaTensor(x, device);
    at::Tensor xla_y = bridge::CreateXlaTensor(y, device);
    at::Tensor xla_result = at::einsum(equation, {xla_x, xla_y});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestAddCMul) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor d = at::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::addcmul(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestAddCDiv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat))) + 1.0;
  at::Tensor d = at::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::addcdiv(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestSize) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(at::size(input, dim), at::size(xla_input, dim));
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSelect) {
  at::Tensor input = at::rand({14, 24, 8}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      at::Tensor output = at::select(input, dim, 4);
      at::Tensor xla_output = at::select(xla_input, dim, 4);
      AllClose(output, xla_output);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestDropout) {
  at::Tensor a = at::rand({17, 21}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor b = at::dropout(xla_a, 0.1, /*train=*/true).cpu();
    double prob =
        static_cast<double>(b.ne(0.0f).sum().item().toDouble()) / a.numel();
    EXPECT_GT(prob, 0.06);
    EXPECT_LT(prob, 0.14);
  });
}

TEST_F(AtenXlaTensorTest, TestSlice) {
  at::Tensor a = at::rand({32, 24, 16}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::slice(xla_a, 1, 0, 16, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestStack) {
  at::Tensor a = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor d = at::stack({a, b, c}, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::stack({xla_a, xla_b, xla_c}, 1);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestCat) {
  at::Tensor a = at::rand({2, 1, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({2, 3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor d = at::cat({a, b, c}, 1);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::cat({xla_a, xla_b, xla_c}, 1);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestUnbind) {
  at::Tensor input = at::rand({4, 3, 7}, at::TensorOptions(at::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<at::Tensor> output = at::unbind(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      std::vector<at::Tensor> xla_output = at::unbind(xla_input, dim);
      ASSERT_EQ(output.size(), xla_output.size());
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], xla_output[i]);
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestRepeat) {
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      at::Tensor input = GetTestTensor(input_size);
      at::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        at::Tensor xla_output = xla_input.repeat(repeats);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestGather) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::empty({3, 3}, at::TensorOptions(at::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    at::Tensor c = at::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
      at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
      at::Tensor xla_c = at::gather(xla_a, 1, xla_b, sparse_grad);
      AllClose(c, xla_c);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestIndexSelect) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::empty({2}, at::TensorOptions(at::kLong));
  b[0] = 0;
  b[1] = 2;
  at::Tensor c0 = at::index_select(a, 0, b);
  at::Tensor c1 = at::index_select(a, 1, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c0 = at::index_select(xla_a, 0, xla_b);
    at::Tensor xla_c1 = at::index_select(xla_a, 1, xla_b);
    AllClose(c0, xla_c0);
    AllClose(c1, xla_c1);
  });
}

TEST_F(AtenXlaTensorTest, TestExpand) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::native::expand(a, {2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::native::expand(xla_a, {2, 3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandBack) {
  at::Tensor a = at::rand({3, 1}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::native::expand(a, {3, 4}, /*implicit=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::native::expand(xla_a, {3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestExpandAs) {
  at::Tensor a = at::rand({3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 3, 4}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::native::expand_as(a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = at::native::expand_as(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenXlaTensorTest, TestBroadcastTensors) {
  at::Tensor a = at::rand({2, 1, 1}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2, 1}, at::TensorOptions(at::kFloat));
  std::vector<at::Tensor> c = at::broadcast_tensors({a, b});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    std::vector<at::Tensor> xla_c = at::broadcast_tensors({xla_a, xla_b});
    ASSERT_EQ(c.size(), xla_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], xla_c[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestOneIndex) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
    at::Tensor xla_result = at::index(xla_params, {xla_indices});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestOneIndexTransfer) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_result = at::index(xla_params, {indices});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleNull) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_null;
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices_0, indices_null, indices_1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
    at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
    at::Tensor xla_result =
        at::index(xla_params, {xla_indices_0, indices_null, xla_indices_1});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailNull) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_null;
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices_0, indices_1, indices_null});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
    at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
    at::Tensor xla_result =
        at::index(xla_params, {xla_indices_0, xla_indices_1, indices_null});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMultiIndexMiddleBroadcast) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 4, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices_0, indices_1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
    at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
    at::Tensor xla_result =
        at::index(xla_params, {xla_indices_0, xla_indices_1});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMultiIndexTailBroadcast) {
  at::Tensor params = at::rand({4, 3, 5, 6, 7}, at::TensorOptions(at::kFloat));
  at::Tensor indices_0 =
      at::randint(-3, 3, {2, 1, 3}, at::TensorOptions(at::kLong));
  at::Tensor indices_1 =
      at::randint(-3, 3, {2, 1}, at::TensorOptions(at::kLong));
  at::Tensor result = at::index(params, {indices_0, indices_1});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices_0 = bridge::CreateXlaTensor(indices_0, device);
    at::Tensor xla_indices_1 = bridge::CreateXlaTensor(indices_1, device);
    at::Tensor xla_result =
        at::index(xla_params, {xla_indices_0, xla_indices_1});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMaskIndex) {
  at::Tensor params = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor indices = at::randint(0, 2, {2, 2}, at::TensorOptions(at::kByte));
  at::Tensor result = at::index(params, {indices});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_params = bridge::CreateXlaTensor(params, device);
    at::Tensor xla_indices = bridge::CreateXlaTensor(indices, device);
    at::Tensor xla_result = at::index(xla_params, {xla_indices});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::relu(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::relu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestLeakyRelu) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  double negative_slope = 0.01;
  at::Tensor output = at::leaky_relu(input, negative_slope);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::leaky_relu(xla_input, negative_slope);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestExp) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::exp(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::exp(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestExpm1) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::expm1(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::expm1(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog2) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log2(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log2(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog10) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log10(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log10(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestLog1p) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::log1p(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::log1p(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErf) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erf(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erf(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfc) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erfc(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erfc(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestErfinv) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::erfinv(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::erfinv(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestSqrt) {
  at::Tensor a = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor b = at::sqrt(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::sqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestRsqrt) {
  at::Tensor a = at::abs(at::rand({2, 2}, at::TensorOptions(at::kFloat)));
  at::Tensor b = at::rsqrt(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::rsqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestReciprocal) {
  at::Tensor a = at::randn({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::reciprocal(a);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::reciprocal(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestPow) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::pow(a, 4.09);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::pow(xla_a, 4.09);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenXlaTensorTest, TestFmod) {
  at::Tensor a = at::rand({2, 2}, at::TensorOptions(at::kFloat)) * 100.0;
  at::Tensor b = at::fmod(a, 2.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = at::fmod(xla_a, 2.0);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestWhere) {
  at::Tensor a = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({3, 3}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::empty({3, 3}, at::TensorOptions(at::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  at::Tensor d = at::where(c, a, b);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    at::Tensor xla_d = at::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenXlaTensorTest, TestThreshold) {
  at::Tensor input = at::rand({2, 1, 4, 6}, at::TensorOptions(at::kFloat));
  float threshold = 0.4;
  float value = 20;
  at::Tensor output = at::threshold(input, threshold, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::threshold(xla_input, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  at::Tensor input =
      at::rand({in_channels, out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, labels}, at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({labels}, at::TensorOptions(at::kFloat));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    at::Tensor output = at::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
      at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
      at::Tensor xla_output =
          at::addmm(xla_bias, xla_input, xla_weight, /*beta=*/beta);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestEmbedding) {
  at::Tensor a = at::rand({32, 4}, at::TensorOptions(at::kFloat));
  at::Tensor i = at::randint(0, 31, {3, 4}, at::TensorOptions(at::kLong));
  at::Tensor b =
      at::embedding(a, i, /*padding_idx=*/0, /*scale_grad_by_freq=*/false,
                    /*sparse=*/false);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_i = bridge::CreateXlaTensor(i, device);
    at::Tensor xla_b = at::embedding(xla_a, xla_i, /*padding_idx=*/0,
                                     /*scale_grad_by_freq=*/false,
                                     /*sparse=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenXlaTensorTest, TestTranspose) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::t(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::t(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestReshape) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = at::reshape(input, {-1, 320});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::reshape(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestView) {
  at::Tensor input = at::rand({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = xla_input.view({-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output = xla_input.view({-1, 320});
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestViewModComplex) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    at::Tensor xla_output2 = xla_input.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestViewOfViewMod) {
  at::Tensor input = at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
  at::Tensor one = at::tensor(1.0, at::TensorOptions(at::kFloat));
  at::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  at::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const Device& device) {
    at::Tensor xinput =
        at::zeros({32, 20, 4, 4}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(xinput, device);
    at::Tensor xla_one = bridge::CreateXlaTensor(one, device);
    at::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    at::Tensor xla_output2 = xla_output1.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenXlaTensorTest, TestLogSoftmax) {
  at::Tensor input = at::rand({5, 3, 4, 2}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = 0; dim < input.dim(); ++dim) {
      at::Tensor output = at::log_softmax(input, dim);
      at::Tensor xla_output = at::log_softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSoftmax) {
  at::Tensor input = at::rand({10, 8, 24, 16}, at::TensorOptions(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    for (int dim = 0; dim < input.dim(); ++dim) {
      at::Tensor output = at::softmax(input, dim);
      at::Tensor xla_output = at::softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestMaxPool2D) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size},
                               /*stride=*/{stride, stride},
                               /*padding=*/{padding, padding},
                               /*dilation=*/{dilation, dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DNonSquare) {
  at::Tensor input = at::rand({1, 64, 112, 112}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output = at::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::max_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size + 1},
                               /*stride=*/{stride, stride + 1},
                               /*padding=*/{padding, padding + 1},
                               /*dilation=*/{dilation, dilation},
                               /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2D) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size},
                               /*stride=*/{stride, stride},
                               /*padding=*/{padding, padding},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DNonSquare) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          at::Tensor output = at::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_output =
                at::avg_pool2d(xla_input,
                               /*kernel_size=*/{kernel_size, kernel_size + 1},
                               /*stride=*/{stride, stride + 1},
                               /*padding=*/{padding, padding + 1},
                               /*ceil_mode=*/ceil_mode,
                               /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2D) {
  at::Tensor input = at::rand({4, 1, 28, 28}, at::TensorOptions(at::kFloat));
  for (int64_t output_size : {7, 8}) {
    at::Tensor output =
        at::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output =
          at::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestConv2D) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias_undef;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::conv2d(input, weight, with_bias ? bias : bias_undef,
                         /*stride=*/{stride, stride},
                         /*padding=*/{padding, padding},
                         /*dilation=*/{dilation, dilation});
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
            at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
            at::Tensor xla_output = at::conv2d(
                xla_input, xla_weight, with_bias ? xla_bias : bias_undef,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation});
            AllClose(output, xla_output);
          });
        }
      };
    }
  }
}

TEST_F(AtenXlaTensorTest, TestConv2DNonSquare) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  at::Tensor input =
      at::rand({4, in_channels, 28, 28}, at::TensorOptions(at::kFloat));
  at::Tensor weight =
      at::rand({out_channels, in_channels, kernel_size, kernel_size},
               at::TensorOptions(at::kFloat));
  at::Tensor bias = at::rand({out_channels}, at::TensorOptions(at::kFloat));
  at::Tensor bias_undef;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          at::Tensor output =
              at::conv2d(input, weight, with_bias ? bias : bias_undef,
                         /*stride=*/{stride, stride + 1},
                         /*padding=*/{padding, padding + 1},
                         /*dilation=*/{dilation, dilation});
          ForEachDevice([&](const Device& device) {
            at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
            at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
            at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
            at::Tensor xla_output = at::conv2d(
                xla_input, xla_weight, with_bias ? xla_bias : bias_undef,
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation});
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestNllLoss) {
  int batch = 3;
  int classes = 5;
  at::Tensor input = GetTestTensor({batch, classes});
  at::Tensor target =
      at::randint(0, classes, {batch}, at::TensorOptions(at::kLong));
  at::Tensor undef_weight;
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    at::Tensor output =
        at::nll_loss(/*self=*/input, /*target=*/target, /*weight=*/undef_weight,
                     /*reduction=*/reduction);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_target = bridge::CreateXlaTensor(target, device);
      at::Tensor xla_output = at::nll_loss(
          /*self=*/xla_input, /*target=*/xla_target, /*weight=*/undef_weight,
          /*reduction=*/reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2D) {
  int num_features = 3;
  at::Tensor input = GetTestTensor({14, num_features, 5, 7});
  at::Tensor weight = GetTestTensor({num_features});
  at::Tensor bias = GetTestTensor({num_features});
  at::Tensor running_mean =
      at::zeros({num_features}, at::TensorOptions(at::kFloat));
  at::Tensor running_var =
      at::ones({num_features}, at::TensorOptions(at::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  for (bool training : {true, false}) {
    at::Tensor output = at::batch_norm(
        /*input=*/input, /*weight=*/weight, /*bias=*/bias,
        /*running_mean=*/running_mean, /*running_var=*/running_var,
        /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_weight = bridge::CreateXlaTensor(weight, device);
      at::Tensor xla_bias = bridge::CreateXlaTensor(bias, device);
      at::Tensor xla_running_mean =
          bridge::CreateXlaTensor(running_mean, device);
      at::Tensor xla_running_var = bridge::CreateXlaTensor(running_var, device);
      at::Tensor xla_output = at::batch_norm(
          /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
          /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDim) {
  at::Tensor input = GetTestTensor({2, 3});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    EXPECT_EQ(input.dim(), xla_input.dim());
  });
}

TEST_F(AtenXlaTensorTest, TestContiguous) {
  at::Tensor input = GetTestTensor({2, 3});
  at::Tensor output = at::native::contiguous(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::native::contiguous(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeAll) {
  at::Tensor input = GetTestTensor({2, 1, 3, 1});
  at::Tensor output = at::squeeze(input);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::squeeze(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const Device& device) {
    at::Tensor input = GetTestTensor({2, 1, 3, 1});
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor output = input.squeeze_();
    at::Tensor xla_output = xla_input.squeeze_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestSqueezeOne) {
  at::Tensor input = GetTestTensor({2, 1, 3, 1});
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor output = at::squeeze(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::squeeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const Device& device) {
      at::Tensor input = GetTestTensor({2, 1, 3, 1});
      at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
      at::Tensor output = input.squeeze_(dim);
      at::Tensor xla_output = xla_input.squeeze_(dim);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestUnsqueeze) {
  at::Tensor input = GetTestTensor({2, 3});
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    at::Tensor output = at::unsqueeze(input, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::unsqueeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestMaskedFill) {
  at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
  at::Tensor mask = at::randint(0, 2, {2, 3}, at::TensorOptions(at::kByte));
  at::Scalar value(42);
  at::Tensor result = at::masked_fill(input, mask, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor xla_result = at::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedFillInPlace) {
  at::Scalar value(42);
  at::Tensor mask = at::randint(0, 2, {2, 3}, at::TensorOptions(at::kByte));
  ForEachDevice([&](const Device& device) {
    at::Tensor input = at::rand({2, 3}, at::TensorOptions(at::kFloat));
    at::Tensor xla_input = bridge::CreateXlaTensor(input.clone(), device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor result = input.masked_fill_(mask, value);
    at::Tensor xla_result = xla_input.masked_fill_(xla_mask, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenXlaTensorTest, TestMaskedFillBroadcast) {
  at::Tensor input = at::rand({2, 5, 4, 3}, at::TensorOptions(at::kFloat));
  at::Tensor mask = at::randint(0, 2, {4, 1}, at::TensorOptions(at::kByte));
  at::Scalar value(42);
  at::Tensor result = at::masked_fill(input, mask, value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_mask = bridge::CreateXlaTensor(mask, device);
    at::Tensor xla_result = at::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestPermute) {
  at::Tensor input = GetTestTensor({2, 3, 4});
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    at::Tensor output = input.permute(dims_permutation);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = xla_input.permute(dims_permutation);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestSplit) {
  at::Tensor input = GetTestTensor({7, 8, 9});
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<at::Tensor> outputs = at::split(input, split_size, dim);
      ForEachDevice([&](const Device& device) {
        at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
        std::vector<at::Tensor> xla_outputs =
            at::split(xla_input, split_size, dim);
        ASSERT_EQ(outputs.size(), xla_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], xla_outputs[i]);
        }
      });
    }
  }
}

TEST_F(AtenXlaTensorTest, TestSplitEmpty) {
  at::Tensor input = GetTestTensor({0});
  int split_size = 0;
  int dim = 0;
  std::vector<at::Tensor> outputs = at::split(input, split_size, dim);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    std::vector<at::Tensor> xla_outputs = at::split(xla_input, split_size, dim);
    ASSERT_EQ(outputs.size(), xla_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], xla_outputs[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestSplitWithSizes) {
  at::Tensor input = GetTestTensor({15, 15, 15});
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<at::Tensor> outputs =
        at::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      std::vector<at::Tensor> xla_outputs =
          at::split_with_sizes(xla_input, {4, 5, 6}, dim);
      ASSERT_EQ(outputs.size(), xla_outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], xla_outputs[i]);
      }
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriu) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriuNonSquare) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size + 1});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  at::Tensor input = GetTestTensor({batch_size, size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::triu(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTril) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrilNonSquare) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size + 1});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  at::Tensor input = GetTestTensor({batch_size, size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::tril(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiag) {
  int size = 7;
  at::Tensor input = GetTestTensor({size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diag(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonal) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diagonal(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalNonSquare) {
  int size = 5;
  at::Tensor input = GetTestTensor({size, size + 1});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output = at::diagonal(input, diagonal);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output = at::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  at::Tensor input = GetTestTensor({batch_size, size, size});
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    at::Tensor output =
        at::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const Device& device) {
      at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
      at::Tensor xla_output =
          at::diagonal(xla_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestBitwiseAnd) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__and__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseAndScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__and__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOr) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__or__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseOrScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__or__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXor) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor rhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_rhs = bridge::CreateXlaTensor(rhs, device);
    at::Tensor xla_result = xla_lhs.__xor__(xla_rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestBitwiseXorScalar) {
  at::Tensor lhs = at::randint(0, std::numeric_limits<int32_t>::max(), {4, 2},
                               at::TensorOptions(at::kInt));
  at::Scalar rhs(123456789);
  at::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_lhs = bridge::CreateXlaTensor(lhs, device);
    at::Tensor xla_result = xla_lhs.__xor__(rhs);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenXlaTensorTest, TestMeshgrid) {
  at::Tensor a = at::rand({3}, at::TensorOptions(at::kFloat));
  at::Tensor b = at::rand({2}, at::TensorOptions(at::kFloat));
  at::Tensor c = at::rand({4}, at::TensorOptions(at::kFloat));
  auto d = at::meshgrid({a, b, c});
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateXlaTensor(a, device);
    at::Tensor xla_b = bridge::CreateXlaTensor(b, device);
    at::Tensor xla_c = bridge::CreateXlaTensor(c, device);
    auto xla_d = at::meshgrid({xla_a, xla_b, xla_c});
    EXPECT_EQ(d.size(), xla_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], xla_d[i]);
    }
  });
}

TEST_F(AtenXlaTensorTest, TestConstantPad) {
  at::Tensor input = at::rand({4, 2, 5}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  at::Tensor output = at::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestConstantPadIncomplete) {
  at::Tensor input = at::rand({4, 2, 5}, at::TensorOptions(at::kFloat));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  at::Tensor output = at::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_input = bridge::CreateXlaTensor(input, device);
    at::Tensor xla_output = at::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenXlaTensorTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
            return at::avg_pool2d(inputs[0],
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const Device& device) {
            TestBackward({GetTestTensor({4, 1, 28, 28})}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const Device& device) {
      TestBackward({GetTestTensor({4, 1, 28, 28})}, device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestConv2DBackward) {
  int in_channels = 3;
  int out_channels = 7;
  int kernel_size = 5;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto testfn =
              [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
            return at::conv2d(inputs[0], inputs[1], inputs[2],
                              /*stride=*/{stride, stride},
                              /*padding=*/{padding, padding},
                              /*dilation=*/{dilation, dilation});
          };

          ForEachDevice([&](const Device& device) {
            at::Tensor bias =
                with_bias ? GetTestTensor({out_channels}) : at::Tensor();
            TestBackward({GetTestTensor({4, in_channels, 32, 32}),
                          GetTestTensor({out_channels, in_channels, kernel_size,
                                         kernel_size}),
                          bias},
                         device, testfn);
          });
        }
      };
    }
  }
}

TEST_F(AtenXlaTensorTest, TestMaxPool2DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
          return at::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const Device& device) {
          TestBackward({GetTestTensor({1, 64, 112, 112})}, device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenXlaTensorTest, TestLogSoftmaxBackward) {
  for (int dim = 0; dim < 4; ++dim) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const Device& device) {
      TestBackward({GetTestTensor({5, 3, 4, 2})}, device, testfn, /*rtol=*/1e-3,
                   /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestReluBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::relu(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({2, 1, 4, 6})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::t(inputs[0]);
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({2, 3})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestAddMatMulBackward) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    ForEachDevice([&](const Device& device) {
      TestBackward(
          {GetTestTensor({labels}), GetTestTensor({in_channels, out_channels}),
           GetTestTensor({out_channels, labels})},
          device, testfn);
    });
  }
}

TEST_F(AtenXlaTensorTest, TestNllLossBackward) {
  int batch = 3;
  int classes = 5;
  at::Tensor input = GetTestTensor({batch, classes});
  at::Tensor target =
      at::randint(0, classes, {batch}, at::TensorOptions(at::kLong));
  at::Tensor undef_weight;
  for (Reduction::Reduction reduction : {Reduction::Mean, Reduction::Sum}) {
    auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
      return at::nll_loss(
          /*self=*/inputs[0], /*target=*/inputs[1], /*weight=*/undef_weight,
          /*reduction=*/reduction);
    };
    ForEachDevice([&](const Device& device) {
      TestBackward({input, target}, device, testfn, /*rtol=*/1e-5,
                   /*atol=*/1e-8, /*inputs_require_grad=*/{true, false});
    });
  }
}

TEST_F(AtenXlaTensorTest, TestViewBackward) {
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const Device& device) {
    TestBackward({GetTestTensor({32, 20, 4, 4})}, device, testfn);
  });
}

TEST_F(AtenXlaTensorTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<at::Tensor>& inputs) -> at::Tensor {
    return at::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  ForEachDevice([&](const Device& device) {
    at::Tensor input = GetTestTensor({14, num_features, 5, 7});
    at::Tensor weight = GetTestTensor({num_features});
    at::Tensor bias = GetTestTensor({num_features});
    at::Tensor running_mean =
        at::zeros({num_features}, at::TensorOptions(at::kFloat));
    at::Tensor running_var =
        at::ones({num_features}, at::TensorOptions(at::kFloat));
    TestBackward({input, weight, bias, running_mean, running_var}, device,
                 testfn,
                 /*rtol=*/1e-3, /*atol=*/1e-4,
                 /*inputs_require_grad=*/{true, true, true, false, false});
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
