#include "torch_xla/csrc/ops/narrow.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, xla::int64 start,
                           xla::int64 length) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return xla::SliceInDim(operands[0], start, start + length, 1, dim);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Narrow::Narrow(const Value& input, xla::int64 dim, xla::int64 start,
               xla::int64 length)
    : Node(ir::OpKind(at::aten::narrow), {input},
           NodeOutputShape(input, dim, start, length),
           /*num_outputs=*/1, xla::util::MHash(dim, start, length)),
      dim_(dim),
      start_(start),
      length_(length) {}

XlaOpVector Narrow::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::SliceInDim(input, start_, start_ + length_, 1, dim_),
                  loctx);
}

std::string Narrow::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", length=" << length_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
