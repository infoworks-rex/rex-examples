
I
PlaceholderPlaceholder*
dtype0* 
shape:���������
G
Placeholder_1Placeholder*
dtype0*
shape:���������
G
Placeholder_2Placeholder*
dtype0*
shape:���������
G
Placeholder_3Placeholder*
dtype0*
shape:���������
2
rnn/RankConst*
value	B :*
dtype0
9
rnn/range/startConst*
value	B :*
dtype0
9
rnn/range/deltaConst*
value	B :*
dtype0
J
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0
H
rnn/concat/values_0Const*
valueB"       *
dtype0
9
rnn/concat/axisConst*
value	B : *
dtype0
e

rnn/concatConcatV2rnn/concat/values_0	rnn/rangernn/concat/axis*
T0*
N*

Tidx0
I
rnn/transpose	TransposePlaceholder
rnn/concat*
Tperm0*
T0
:
	rnn/ShapeShapernn/transpose*
T0*
out_type0
E
rnn/strided_slice/stackConst*
valueB:*
dtype0
G
rnn/strided_slice/stack_1Const*
valueB:*
dtype0
G
rnn/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
N
$rnn/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
|
 rnn/LSTMCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice$rnn/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
I
rnn/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
K
!rnn/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
rnn/LSTMCellZeroState/concatConcatV2 rnn/LSTMCellZeroState/ExpandDimsrnn/LSTMCellZeroState/Const!rnn/LSTMCellZeroState/concat/axis*

Tidx0*
T0*
N
N
!rnn/LSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    

rnn/LSTMCellZeroState/zerosFillrnn/LSTMCellZeroState/concat!rnn/LSTMCellZeroState/zeros/Const*
T0*

index_type0
P
&rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0
�
"rnn/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0
K
rnn/LSTMCellZeroState/Const_1Const*
valueB:x*
dtype0
P
&rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
"rnn/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
K
rnn/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
M
#rnn/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
rnn/LSTMCellZeroState/concat_1ConcatV2"rnn/LSTMCellZeroState/ExpandDims_2rnn/LSTMCellZeroState/Const_2#rnn/LSTMCellZeroState/concat_1/axis*
N*

Tidx0*
T0
P
#rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn/LSTMCellZeroState/zeros_1Fillrnn/LSTMCellZeroState/concat_1#rnn/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
P
&rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
"rnn/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
K
rnn/LSTMCellZeroState/Const_3Const*
valueB:x*
dtype0
<
rnn/Shape_1Shapernn/transpose*
T0*
out_type0
G
rnn/strided_slice_1/stackConst*
valueB: *
dtype0
I
rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0
I
rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
rnn/strided_slice_1StridedSlicernn/Shape_1rnn/strided_slice_1/stackrnn/strided_slice_1/stack_1rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
<
rnn/Shape_2Shapernn/transpose*
T0*
out_type0
G
rnn/strided_slice_2/stackConst*
valueB:*
dtype0
I
rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0
I
rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
rnn/strided_slice_2StridedSlicernn/Shape_2rnn/strided_slice_2/stackrnn/strided_slice_2/stack_1rnn/strided_slice_2/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
<
rnn/ExpandDims/dimConst*
dtype0*
value	B : 
Z
rnn/ExpandDims
ExpandDimsrnn/strided_slice_2rnn/ExpandDims/dim*
T0*

Tdim0
7
	rnn/ConstConst*
valueB:x*
dtype0
;
rnn/concat_1/axisConst*
value	B : *
dtype0
d
rnn/concat_1ConcatV2rnn/ExpandDims	rnn/Constrnn/concat_1/axis*

Tidx0*
T0*
N
<
rnn/zeros/ConstConst*
valueB
 *    *
dtype0
K
	rnn/zerosFillrnn/concat_1rnn/zeros/Const*
T0*

index_type0
2
rnn/timeConst*
value	B : *
dtype0
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice_1*/
tensor_array_namernn/dynamic_rnn/output_0*
dtype0*$
element_shape:���������x*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
�
rnn/TensorArray_1TensorArrayV3rnn/strided_slice_1*.
tensor_array_namernn/dynamic_rnn/input_0*
dtype0*$
element_shape:���������*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
M
rnn/TensorArrayUnstack/ShapeShapernn/transpose*
T0*
out_type0
X
*rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
Z
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
Z
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
L
"rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
L
"rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/rangernn/transposernn/TensorArray_1:1*
T0* 
_class
loc:@rnn/transpose
7
rnn/Maximum/xConst*
value	B :*
dtype0
C
rnn/MaximumMaximumrnn/Maximum/xrnn/strided_slice_1*
T0
A
rnn/MinimumMinimumrnn/strided_slice_1rnn/Maximum*
T0
E
rnn/while/iteration_counterConst*
value	B : *
dtype0
�
rnn/while/EnterEnterrnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_1Enterrnn/time*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_2Enterrnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_3Enterrnn/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_4Enterrnn/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
T
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
N
Z
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
N*
T0
Z
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N
Z
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N
Z
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0*
N
F
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0
�
rnn/while/Less/EnterEnterrnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
L
rnn/while/Less_1Lessrnn/while/Merge_1rnn/while/Less_1/Enter*
T0
�
rnn/while/Less_1/EnterEnterrnn/Minimum*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
D
rnn/while/LogicalAnd
LogicalAndrnn/while/Lessrnn/while/Less_1
4
rnn/while/LoopCondLoopCondrnn/while/LogicalAnd
l
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
T0*"
_class
loc:@rnn/while/Merge
r
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_1
r
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_2
r
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_3
r
rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_4
;
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0
?
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0
?
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0
?
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0
?
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0
N
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
dtype0
B
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0
�
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity_1#rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
0rnn/LSTM/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@rnn/LSTM/kernel*
valueB"~   �  

.rnn/LSTM/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@rnn/LSTM/kernel*
valueB
 *��˽

.rnn/LSTM/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@rnn/LSTM/kernel*
valueB
 *���=*
dtype0
�
8rnn/LSTM/kernel/Initializer/random_uniform/RandomUniformRandomUniform0rnn/LSTM/kernel/Initializer/random_uniform/shape*
dtype0*
seed2h*
seed�*
T0*"
_class
loc:@rnn/LSTM/kernel
�
.rnn/LSTM/kernel/Initializer/random_uniform/subSub.rnn/LSTM/kernel/Initializer/random_uniform/max.rnn/LSTM/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@rnn/LSTM/kernel
�
.rnn/LSTM/kernel/Initializer/random_uniform/mulMul8rnn/LSTM/kernel/Initializer/random_uniform/RandomUniform.rnn/LSTM/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@rnn/LSTM/kernel
�
*rnn/LSTM/kernel/Initializer/random_uniformAdd.rnn/LSTM/kernel/Initializer/random_uniform/mul.rnn/LSTM/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@rnn/LSTM/kernel
�
rnn/LSTM/kernel
VariableV2*
dtype0*
	container *
shape:	~�*
shared_name *"
_class
loc:@rnn/LSTM/kernel
�
rnn/LSTM/kernel/AssignAssignrnn/LSTM/kernel*rnn/LSTM/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel*
validate_shape(
:
rnn/LSTM/kernel/readIdentityrnn/LSTM/kernel*
T0
s
rnn/LSTM/bias/Initializer/zerosConst* 
_class
loc:@rnn/LSTM/bias*
valueB�*    *
dtype0
�
rnn/LSTM/bias
VariableV2* 
_class
loc:@rnn/LSTM/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM/bias/AssignAssignrnn/LSTM/biasrnn/LSTM/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias*
validate_shape(
6
rnn/LSTM/bias/readIdentityrnn/LSTM/bias*
T0
Y
rnn/while/LSTM/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0
�
rnn/while/LSTM/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_4rnn/while/LSTM/concat/axis*
N*

Tidx0*
T0
�
rnn/while/LSTM/MatMulMatMulrnn/while/LSTM/concatrnn/while/LSTM/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
rnn/while/LSTM/MatMul/EnterEnterrnn/LSTM/kernel/read*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
v
rnn/while/LSTM/BiasAddBiasAddrnn/while/LSTM/MatMulrnn/while/LSTM/BiasAdd/Enter*
T0*
data_formatNHWC
�
rnn/while/LSTM/BiasAdd/EnterEnterrnn/LSTM/bias/read*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
S
rnn/while/LSTM/ConstConst^rnn/while/Identity*
dtype0*
value	B :
]
rnn/while/LSTM/split/split_dimConst^rnn/while/Identity*
value	B :*
dtype0
o
rnn/while/LSTM/splitSplitrnn/while/LSTM/split/split_dimrnn/while/LSTM/BiasAdd*
T0*
	num_split
V
rnn/while/LSTM/add/yConst^rnn/while/Identity*
valueB
 *  �?*
dtype0
P
rnn/while/LSTM/addAddrnn/while/LSTM/split:2rnn/while/LSTM/add/y*
T0
>
rnn/while/LSTM/SigmoidSigmoidrnn/while/LSTM/add*
T0
P
rnn/while/LSTM/mulMulrnn/while/LSTM/Sigmoidrnn/while/Identity_3*
T0
B
rnn/while/LSTM/Sigmoid_1Sigmoidrnn/while/LSTM/split*
T0
<
rnn/while/LSTM/TanhTanhrnn/while/LSTM/split:1*
T0
S
rnn/while/LSTM/mul_1Mulrnn/while/LSTM/Sigmoid_1rnn/while/LSTM/Tanh*
T0
N
rnn/while/LSTM/add_1Addrnn/while/LSTM/mulrnn/while/LSTM/mul_1*
T0
D
rnn/while/LSTM/Sigmoid_2Sigmoidrnn/while/LSTM/split:3*
T0
<
rnn/while/LSTM/Tanh_1Tanhrnn/while/LSTM/add_1*
T0
U
rnn/while/LSTM/mul_2Mulrnn/while/LSTM/Sigmoid_2rnn/while/LSTM/Tanh_1*
T0
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity_1rnn/while/LSTM/mul_2rnn/while/Identity_2*
T0*'
_class
loc:@rnn/while/LSTM/mul_2
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
T0*'
_class
loc:@rnn/while/LSTM/mul_2*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
P
rnn/while/add_1/yConst^rnn/while/Identity*
dtype0*
value	B :
H
rnn/while/add_1Addrnn/while/Identity_1rnn/while/add_1/y*
T0
@
rnn/while/NextIterationNextIterationrnn/while/add*
T0
D
rnn/while/NextIteration_1NextIterationrnn/while/add_1*
T0
b
rnn/while/NextIteration_2NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
I
rnn/while/NextIteration_3NextIterationrnn/while/LSTM/add_1*
T0
I
rnn/while/NextIteration_4NextIterationrnn/while/LSTM/mul_2*
T0
1
rnn/while/ExitExitrnn/while/Switch*
T0
5
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0
5
rnn/while/Exit_2Exitrnn/while/Switch_2*
T0
5
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0
5
rnn/while/Exit_4Exitrnn/while/Switch_4*
T0
�
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_2*"
_class
loc:@rnn/TensorArray
n
 rnn/TensorArrayStack/range/startConst*"
_class
loc:@rnn/TensorArray*
value	B : *
dtype0
n
 rnn/TensorArrayStack/range/deltaConst*"
_class
loc:@rnn/TensorArray*
value	B :*
dtype0
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*"
_class
loc:@rnn/TensorArray*

Tidx0
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_2*$
element_shape:���������x*"
_class
loc:@rnn/TensorArray*
dtype0
9
rnn/Const_1Const*
dtype0*
valueB:x
4

rnn/Rank_1Const*
dtype0*
value	B :
;
rnn/range_1/startConst*
dtype0*
value	B :
;
rnn/range_1/deltaConst*
value	B :*
dtype0
R
rnn/range_1Rangernn/range_1/start
rnn/Rank_1rnn/range_1/delta*

Tidx0
J
rnn/concat_2/values_0Const*
valueB"       *
dtype0
;
rnn/concat_2/axisConst*
value	B : *
dtype0
m
rnn/concat_2ConcatV2rnn/concat_2/values_0rnn/range_1rnn/concat_2/axis*
T0*
N*

Tidx0
j
rnn/transpose_1	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_2*
Tperm0*
T0
4

rnn_1/RankConst*
value	B :*
dtype0
;
rnn_1/range/startConst*
value	B :*
dtype0
;
rnn_1/range/deltaConst*
value	B :*
dtype0
R
rnn_1/rangeRangernn_1/range/start
rnn_1/Rankrnn_1/range/delta*

Tidx0
J
rnn_1/concat/values_0Const*
valueB"       *
dtype0
;
rnn_1/concat/axisConst*
dtype0*
value	B : 
m
rnn_1/concatConcatV2rnn_1/concat/values_0rnn_1/rangernn_1/concat/axis*
T0*
N*

Tidx0
Q
rnn_1/transpose	Transposernn/transpose_1rnn_1/concat*
Tperm0*
T0
>
rnn_1/ShapeShapernn_1/transpose*
T0*
out_type0
G
rnn_1/strided_slice/stackConst*
valueB:*
dtype0
I
rnn_1/strided_slice/stack_1Const*
valueB:*
dtype0
I
rnn_1/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_1/strided_sliceStridedSlicernn_1/Shapernn_1/strided_slice/stackrnn_1/strided_slice/stack_1rnn_1/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
P
&rnn_1/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
�
"rnn_1/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_1/strided_slice&rnn_1/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
K
rnn_1/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
M
#rnn_1/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
rnn_1/LSTMCellZeroState/concatConcatV2"rnn_1/LSTMCellZeroState/ExpandDimsrnn_1/LSTMCellZeroState/Const#rnn_1/LSTMCellZeroState/concat/axis*

Tidx0*
T0*
N
P
#rnn_1/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
rnn_1/LSTMCellZeroState/zerosFillrnn_1/LSTMCellZeroState/concat#rnn_1/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_1/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : 
�
$rnn_1/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_1/strided_slice(rnn_1/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0
M
rnn_1/LSTMCellZeroState/Const_1Const*
dtype0*
valueB:x
R
(rnn_1/LSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : 
�
$rnn_1/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_1/strided_slice(rnn_1/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
M
rnn_1/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
O
%rnn_1/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
 rnn_1/LSTMCellZeroState/concat_1ConcatV2$rnn_1/LSTMCellZeroState/ExpandDims_2rnn_1/LSTMCellZeroState/Const_2%rnn_1/LSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0
R
%rnn_1/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn_1/LSTMCellZeroState/zeros_1Fill rnn_1/LSTMCellZeroState/concat_1%rnn_1/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_1/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
$rnn_1/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_1/strided_slice(rnn_1/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0
M
rnn_1/LSTMCellZeroState/Const_3Const*
dtype0*
valueB:x
@
rnn_1/Shape_1Shapernn_1/transpose*
T0*
out_type0
I
rnn_1/strided_slice_1/stackConst*
valueB: *
dtype0
K
rnn_1/strided_slice_1/stack_1Const*
valueB:*
dtype0
K
rnn_1/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1rnn_1/strided_slice_1/stackrnn_1/strided_slice_1/stack_1rnn_1/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
@
rnn_1/Shape_2Shapernn_1/transpose*
T0*
out_type0
I
rnn_1/strided_slice_2/stackConst*
valueB:*
dtype0
K
rnn_1/strided_slice_2/stack_1Const*
valueB:*
dtype0
K
rnn_1/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
rnn_1/strided_slice_2StridedSlicernn_1/Shape_2rnn_1/strided_slice_2/stackrnn_1/strided_slice_2/stack_1rnn_1/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
>
rnn_1/ExpandDims/dimConst*
value	B : *
dtype0
`
rnn_1/ExpandDims
ExpandDimsrnn_1/strided_slice_2rnn_1/ExpandDims/dim*

Tdim0*
T0
9
rnn_1/ConstConst*
valueB:x*
dtype0
=
rnn_1/concat_1/axisConst*
value	B : *
dtype0
l
rnn_1/concat_1ConcatV2rnn_1/ExpandDimsrnn_1/Constrnn_1/concat_1/axis*
T0*
N*

Tidx0
>
rnn_1/zeros/ConstConst*
valueB
 *    *
dtype0
Q
rnn_1/zerosFillrnn_1/concat_1rnn_1/zeros/Const*
T0*

index_type0
4

rnn_1/timeConst*
value	B : *
dtype0
�
rnn_1/TensorArrayTensorArrayV3rnn_1/strided_slice_1*1
tensor_array_namernn_1/dynamic_rnn/output_0*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
rnn_1/TensorArray_1TensorArrayV3rnn_1/strided_slice_1*0
tensor_array_namernn_1/dynamic_rnn/input_0*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
Q
rnn_1/TensorArrayUnstack/ShapeShapernn_1/transpose*
T0*
out_type0
Z
,rnn_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
\
.rnn_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
\
.rnn_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
&rnn_1/TensorArrayUnstack/strided_sliceStridedSlicernn_1/TensorArrayUnstack/Shape,rnn_1/TensorArrayUnstack/strided_slice/stack.rnn_1/TensorArrayUnstack/strided_slice/stack_1.rnn_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
N
$rnn_1/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
N
$rnn_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn_1/TensorArrayUnstack/rangeRange$rnn_1/TensorArrayUnstack/range/start&rnn_1/TensorArrayUnstack/strided_slice$rnn_1/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_1/TensorArray_1rnn_1/TensorArrayUnstack/rangernn_1/transposernn_1/TensorArray_1:1*
T0*"
_class
loc:@rnn_1/transpose
9
rnn_1/Maximum/xConst*
value	B :*
dtype0
I
rnn_1/MaximumMaximumrnn_1/Maximum/xrnn_1/strided_slice_1*
T0
G
rnn_1/MinimumMinimumrnn_1/strided_slice_1rnn_1/Maximum*
T0
G
rnn_1/while/iteration_counterConst*
value	B : *
dtype0
�
rnn_1/while/EnterEnterrnn_1/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_1/while/while_context
�
rnn_1/while/Enter_1Enter
rnn_1/time*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_1/while/while_context
�
rnn_1/while/Enter_2Enterrnn_1/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_1/while/while_context
�
rnn_1/while/Enter_3Enterrnn_1/LSTMCellZeroState/zeros*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant( 
�
rnn_1/while/Enter_4Enterrnn_1/LSTMCellZeroState/zeros_1*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant( 
Z
rnn_1/while/MergeMergernn_1/while/Enterrnn_1/while/NextIteration*
T0*
N
`
rnn_1/while/Merge_1Mergernn_1/while/Enter_1rnn_1/while/NextIteration_1*
T0*
N
`
rnn_1/while/Merge_2Mergernn_1/while/Enter_2rnn_1/while/NextIteration_2*
T0*
N
`
rnn_1/while/Merge_3Mergernn_1/while/Enter_3rnn_1/while/NextIteration_3*
N*
T0
`
rnn_1/while/Merge_4Mergernn_1/while/Enter_4rnn_1/while/NextIteration_4*
T0*
N
L
rnn_1/while/LessLessrnn_1/while/Mergernn_1/while/Less/Enter*
T0
�
rnn_1/while/Less/EnterEnterrnn_1/strided_slice_1*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
R
rnn_1/while/Less_1Lessrnn_1/while/Merge_1rnn_1/while/Less_1/Enter*
T0
�
rnn_1/while/Less_1/EnterEnterrnn_1/Minimum*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
J
rnn_1/while/LogicalAnd
LogicalAndrnn_1/while/Lessrnn_1/while/Less_1
8
rnn_1/while/LoopCondLoopCondrnn_1/while/LogicalAnd
t
rnn_1/while/SwitchSwitchrnn_1/while/Mergernn_1/while/LoopCond*
T0*$
_class
loc:@rnn_1/while/Merge
z
rnn_1/while/Switch_1Switchrnn_1/while/Merge_1rnn_1/while/LoopCond*
T0*&
_class
loc:@rnn_1/while/Merge_1
z
rnn_1/while/Switch_2Switchrnn_1/while/Merge_2rnn_1/while/LoopCond*
T0*&
_class
loc:@rnn_1/while/Merge_2
z
rnn_1/while/Switch_3Switchrnn_1/while/Merge_3rnn_1/while/LoopCond*
T0*&
_class
loc:@rnn_1/while/Merge_3
z
rnn_1/while/Switch_4Switchrnn_1/while/Merge_4rnn_1/while/LoopCond*
T0*&
_class
loc:@rnn_1/while/Merge_4
?
rnn_1/while/IdentityIdentityrnn_1/while/Switch:1*
T0
C
rnn_1/while/Identity_1Identityrnn_1/while/Switch_1:1*
T0
C
rnn_1/while/Identity_2Identityrnn_1/while/Switch_2:1*
T0
C
rnn_1/while/Identity_3Identityrnn_1/while/Switch_3:1*
T0
C
rnn_1/while/Identity_4Identityrnn_1/while/Switch_4:1*
T0
R
rnn_1/while/add/yConst^rnn_1/while/Identity*
value	B :*
dtype0
H
rnn_1/while/addAddrnn_1/while/Identityrnn_1/while/add/y*
T0
�
rnn_1/while/TensorArrayReadV3TensorArrayReadV3#rnn_1/while/TensorArrayReadV3/Enterrnn_1/while/Identity_1%rnn_1/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_1/while/TensorArrayReadV3/EnterEnterrnn_1/TensorArray_1*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
�
%rnn_1/while/TensorArrayReadV3/Enter_1Enter@rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
2rnn/LSTM_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@rnn/LSTM_1/kernel*
valueB"�   �  
�
0rnn/LSTM_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@rnn/LSTM_1/kernel*
valueB
 *����*
dtype0
�
0rnn/LSTM_1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@rnn/LSTM_1/kernel*
valueB
 *���=*
dtype0
�
:rnn/LSTM_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2�*
seed�*
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
0rnn/LSTM_1/kernel/Initializer/random_uniform/subSub0rnn/LSTM_1/kernel/Initializer/random_uniform/max0rnn/LSTM_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
0rnn/LSTM_1/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_1/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
,rnn/LSTM_1/kernel/Initializer/random_uniformAdd0rnn/LSTM_1/kernel/Initializer/random_uniform/mul0rnn/LSTM_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
rnn/LSTM_1/kernel
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_1/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_1/kernel/AssignAssignrnn/LSTM_1/kernel,rnn/LSTM_1/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_1/kernel*
validate_shape(
>
rnn/LSTM_1/kernel/readIdentityrnn/LSTM_1/kernel*
T0
w
!rnn/LSTM_1/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@rnn/LSTM_1/bias*
valueB�*    
�
rnn/LSTM_1/bias
VariableV2*
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_1/bias*
dtype0*
	container 
�
rnn/LSTM_1/bias/AssignAssignrnn/LSTM_1/bias!rnn/LSTM_1/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(
:
rnn/LSTM_1/bias/readIdentityrnn/LSTM_1/bias*
T0
_
rnn_1/while/LSTM_1/concat/axisConst^rnn_1/while/Identity*
value	B :*
dtype0
�
rnn_1/while/LSTM_1/concatConcatV2rnn_1/while/TensorArrayReadV3rnn_1/while/Identity_4rnn_1/while/LSTM_1/concat/axis*
T0*
N*

Tidx0
�
rnn_1/while/LSTM_1/MatMulMatMulrnn_1/while/LSTM_1/concatrnn_1/while/LSTM_1/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
rnn_1/while/LSTM_1/MatMul/EnterEnterrnn/LSTM_1/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
rnn_1/while/LSTM_1/BiasAddBiasAddrnn_1/while/LSTM_1/MatMul rnn_1/while/LSTM_1/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn_1/while/LSTM_1/BiasAdd/EnterEnterrnn/LSTM_1/bias/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
Y
rnn_1/while/LSTM_1/ConstConst^rnn_1/while/Identity*
value	B :*
dtype0
c
"rnn_1/while/LSTM_1/split/split_dimConst^rnn_1/while/Identity*
value	B :*
dtype0
{
rnn_1/while/LSTM_1/splitSplit"rnn_1/while/LSTM_1/split/split_dimrnn_1/while/LSTM_1/BiasAdd*
T0*
	num_split
\
rnn_1/while/LSTM_1/add/yConst^rnn_1/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_1/while/LSTM_1/addAddrnn_1/while/LSTM_1/split:2rnn_1/while/LSTM_1/add/y*
T0
F
rnn_1/while/LSTM_1/SigmoidSigmoidrnn_1/while/LSTM_1/add*
T0
Z
rnn_1/while/LSTM_1/mulMulrnn_1/while/LSTM_1/Sigmoidrnn_1/while/Identity_3*
T0
J
rnn_1/while/LSTM_1/Sigmoid_1Sigmoidrnn_1/while/LSTM_1/split*
T0
D
rnn_1/while/LSTM_1/TanhTanhrnn_1/while/LSTM_1/split:1*
T0
_
rnn_1/while/LSTM_1/mul_1Mulrnn_1/while/LSTM_1/Sigmoid_1rnn_1/while/LSTM_1/Tanh*
T0
Z
rnn_1/while/LSTM_1/add_1Addrnn_1/while/LSTM_1/mulrnn_1/while/LSTM_1/mul_1*
T0
L
rnn_1/while/LSTM_1/Sigmoid_2Sigmoidrnn_1/while/LSTM_1/split:3*
T0
D
rnn_1/while/LSTM_1/Tanh_1Tanhrnn_1/while/LSTM_1/add_1*
T0
a
rnn_1/while/LSTM_1/mul_2Mulrnn_1/while/LSTM_1/Sigmoid_2rnn_1/while/LSTM_1/Tanh_1*
T0
�
/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_1/while/Identity_1rnn_1/while/LSTM_1/mul_2rnn_1/while/Identity_2*
T0*+
_class!
loc:@rnn_1/while/LSTM_1/mul_2
�
5rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_1/TensorArray*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*+
_class!
loc:@rnn_1/while/LSTM_1/mul_2*
is_constant(
T
rnn_1/while/add_1/yConst^rnn_1/while/Identity*
value	B :*
dtype0
N
rnn_1/while/add_1Addrnn_1/while/Identity_1rnn_1/while/add_1/y*
T0
D
rnn_1/while/NextIterationNextIterationrnn_1/while/add*
T0
H
rnn_1/while/NextIteration_1NextIterationrnn_1/while/add_1*
T0
f
rnn_1/while/NextIteration_2NextIteration/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_1/while/NextIteration_3NextIterationrnn_1/while/LSTM_1/add_1*
T0
O
rnn_1/while/NextIteration_4NextIterationrnn_1/while/LSTM_1/mul_2*
T0
5
rnn_1/while/ExitExitrnn_1/while/Switch*
T0
9
rnn_1/while/Exit_1Exitrnn_1/while/Switch_1*
T0
9
rnn_1/while/Exit_2Exitrnn_1/while/Switch_2*
T0
9
rnn_1/while/Exit_3Exitrnn_1/while/Switch_3*
T0
9
rnn_1/while/Exit_4Exitrnn_1/while/Switch_4*
T0
�
(rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_1/TensorArrayrnn_1/while/Exit_2*$
_class
loc:@rnn_1/TensorArray
r
"rnn_1/TensorArrayStack/range/startConst*$
_class
loc:@rnn_1/TensorArray*
value	B : *
dtype0
r
"rnn_1/TensorArrayStack/range/deltaConst*$
_class
loc:@rnn_1/TensorArray*
value	B :*
dtype0
�
rnn_1/TensorArrayStack/rangeRange"rnn_1/TensorArrayStack/range/start(rnn_1/TensorArrayStack/TensorArraySizeV3"rnn_1/TensorArrayStack/range/delta*$
_class
loc:@rnn_1/TensorArray*

Tidx0
�
*rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_1/TensorArrayrnn_1/TensorArrayStack/rangernn_1/while/Exit_2*$
element_shape:���������x*$
_class
loc:@rnn_1/TensorArray*
dtype0
;
rnn_1/Const_1Const*
dtype0*
valueB:x
6
rnn_1/Rank_1Const*
value	B :*
dtype0
=
rnn_1/range_1/startConst*
value	B :*
dtype0
=
rnn_1/range_1/deltaConst*
value	B :*
dtype0
Z
rnn_1/range_1Rangernn_1/range_1/startrnn_1/Rank_1rnn_1/range_1/delta*

Tidx0
L
rnn_1/concat_2/values_0Const*
valueB"       *
dtype0
=
rnn_1/concat_2/axisConst*
dtype0*
value	B : 
u
rnn_1/concat_2ConcatV2rnn_1/concat_2/values_0rnn_1/range_1rnn_1/concat_2/axis*

Tidx0*
T0*
N
p
rnn_1/transpose_1	Transpose*rnn_1/TensorArrayStack/TensorArrayGatherV3rnn_1/concat_2*
Tperm0*
T0
4

rnn_2/RankConst*
value	B :*
dtype0
;
rnn_2/range/startConst*
value	B :*
dtype0
;
rnn_2/range/deltaConst*
value	B :*
dtype0
R
rnn_2/rangeRangernn_2/range/start
rnn_2/Rankrnn_2/range/delta*

Tidx0
J
rnn_2/concat/values_0Const*
valueB"       *
dtype0
;
rnn_2/concat/axisConst*
value	B : *
dtype0
m
rnn_2/concatConcatV2rnn_2/concat/values_0rnn_2/rangernn_2/concat/axis*
N*

Tidx0*
T0
S
rnn_2/transpose	Transposernn_1/transpose_1rnn_2/concat*
T0*
Tperm0
>
rnn_2/ShapeShapernn_2/transpose*
T0*
out_type0
G
rnn_2/strided_slice/stackConst*
dtype0*
valueB:
I
rnn_2/strided_slice/stack_1Const*
dtype0*
valueB:
I
rnn_2/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_2/strided_sliceStridedSlicernn_2/Shapernn_2/strided_slice/stackrnn_2/strided_slice/stack_1rnn_2/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
P
&rnn_2/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
"rnn_2/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_2/strided_slice&rnn_2/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
K
rnn_2/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
M
#rnn_2/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0
�
rnn_2/LSTMCellZeroState/concatConcatV2"rnn_2/LSTMCellZeroState/ExpandDimsrnn_2/LSTMCellZeroState/Const#rnn_2/LSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
P
#rnn_2/LSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    
�
rnn_2/LSTMCellZeroState/zerosFillrnn_2/LSTMCellZeroState/concat#rnn_2/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_2/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : 
�
$rnn_2/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_2/strided_slice(rnn_2/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0
M
rnn_2/LSTMCellZeroState/Const_1Const*
valueB:x*
dtype0
R
(rnn_2/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
$rnn_2/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_2/strided_slice(rnn_2/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
M
rnn_2/LSTMCellZeroState/Const_2Const*
dtype0*
valueB:x
O
%rnn_2/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
 rnn_2/LSTMCellZeroState/concat_1ConcatV2$rnn_2/LSTMCellZeroState/ExpandDims_2rnn_2/LSTMCellZeroState/Const_2%rnn_2/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N
R
%rnn_2/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    
�
rnn_2/LSTMCellZeroState/zeros_1Fill rnn_2/LSTMCellZeroState/concat_1%rnn_2/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_2/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
$rnn_2/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_2/strided_slice(rnn_2/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0
M
rnn_2/LSTMCellZeroState/Const_3Const*
dtype0*
valueB:x
@
rnn_2/Shape_1Shapernn_2/transpose*
T0*
out_type0
I
rnn_2/strided_slice_1/stackConst*
valueB: *
dtype0
K
rnn_2/strided_slice_1/stack_1Const*
valueB:*
dtype0
K
rnn_2/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1rnn_2/strided_slice_1/stackrnn_2/strided_slice_1/stack_1rnn_2/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
@
rnn_2/Shape_2Shapernn_2/transpose*
T0*
out_type0
I
rnn_2/strided_slice_2/stackConst*
valueB:*
dtype0
K
rnn_2/strided_slice_2/stack_1Const*
valueB:*
dtype0
K
rnn_2/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
rnn_2/strided_slice_2StridedSlicernn_2/Shape_2rnn_2/strided_slice_2/stackrnn_2/strided_slice_2/stack_1rnn_2/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
>
rnn_2/ExpandDims/dimConst*
dtype0*
value	B : 
`
rnn_2/ExpandDims
ExpandDimsrnn_2/strided_slice_2rnn_2/ExpandDims/dim*
T0*

Tdim0
9
rnn_2/ConstConst*
valueB:x*
dtype0
=
rnn_2/concat_1/axisConst*
value	B : *
dtype0
l
rnn_2/concat_1ConcatV2rnn_2/ExpandDimsrnn_2/Constrnn_2/concat_1/axis*

Tidx0*
T0*
N
>
rnn_2/zeros/ConstConst*
valueB
 *    *
dtype0
Q
rnn_2/zerosFillrnn_2/concat_1rnn_2/zeros/Const*
T0*

index_type0
4

rnn_2/timeConst*
value	B : *
dtype0
�
rnn_2/TensorArrayTensorArrayV3rnn_2/strided_slice_1*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*1
tensor_array_namernn_2/dynamic_rnn/output_0*
dtype0
�
rnn_2/TensorArray_1TensorArrayV3rnn_2/strided_slice_1*$
element_shape:���������x*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*0
tensor_array_namernn_2/dynamic_rnn/input_0*
dtype0
Q
rnn_2/TensorArrayUnstack/ShapeShapernn_2/transpose*
T0*
out_type0
Z
,rnn_2/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
\
.rnn_2/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
\
.rnn_2/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
&rnn_2/TensorArrayUnstack/strided_sliceStridedSlicernn_2/TensorArrayUnstack/Shape,rnn_2/TensorArrayUnstack/strided_slice/stack.rnn_2/TensorArrayUnstack/strided_slice/stack_1.rnn_2/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
N
$rnn_2/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
N
$rnn_2/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
rnn_2/TensorArrayUnstack/rangeRange$rnn_2/TensorArrayUnstack/range/start&rnn_2/TensorArrayUnstack/strided_slice$rnn_2/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_2/TensorArray_1rnn_2/TensorArrayUnstack/rangernn_2/transposernn_2/TensorArray_1:1*
T0*"
_class
loc:@rnn_2/transpose
9
rnn_2/Maximum/xConst*
value	B :*
dtype0
I
rnn_2/MaximumMaximumrnn_2/Maximum/xrnn_2/strided_slice_1*
T0
G
rnn_2/MinimumMinimumrnn_2/strided_slice_1rnn_2/Maximum*
T0
G
rnn_2/while/iteration_counterConst*
dtype0*
value	B : 
�
rnn_2/while/EnterEnterrnn_2/while/iteration_counter*
parallel_iterations *)

frame_namernn_2/while/while_context*
T0*
is_constant( 
�
rnn_2/while/Enter_1Enter
rnn_2/time*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_2/while/while_context
�
rnn_2/while/Enter_2Enterrnn_2/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_2/while/while_context
�
rnn_2/while/Enter_3Enterrnn_2/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_2/while/while_context
�
rnn_2/while/Enter_4Enterrnn_2/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_2/while/while_context
Z
rnn_2/while/MergeMergernn_2/while/Enterrnn_2/while/NextIteration*
T0*
N
`
rnn_2/while/Merge_1Mergernn_2/while/Enter_1rnn_2/while/NextIteration_1*
T0*
N
`
rnn_2/while/Merge_2Mergernn_2/while/Enter_2rnn_2/while/NextIteration_2*
T0*
N
`
rnn_2/while/Merge_3Mergernn_2/while/Enter_3rnn_2/while/NextIteration_3*
T0*
N
`
rnn_2/while/Merge_4Mergernn_2/while/Enter_4rnn_2/while/NextIteration_4*
N*
T0
L
rnn_2/while/LessLessrnn_2/while/Mergernn_2/while/Less/Enter*
T0
�
rnn_2/while/Less/EnterEnterrnn_2/strided_slice_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
R
rnn_2/while/Less_1Lessrnn_2/while/Merge_1rnn_2/while/Less_1/Enter*
T0
�
rnn_2/while/Less_1/EnterEnterrnn_2/Minimum*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
J
rnn_2/while/LogicalAnd
LogicalAndrnn_2/while/Lessrnn_2/while/Less_1
8
rnn_2/while/LoopCondLoopCondrnn_2/while/LogicalAnd
t
rnn_2/while/SwitchSwitchrnn_2/while/Mergernn_2/while/LoopCond*
T0*$
_class
loc:@rnn_2/while/Merge
z
rnn_2/while/Switch_1Switchrnn_2/while/Merge_1rnn_2/while/LoopCond*
T0*&
_class
loc:@rnn_2/while/Merge_1
z
rnn_2/while/Switch_2Switchrnn_2/while/Merge_2rnn_2/while/LoopCond*
T0*&
_class
loc:@rnn_2/while/Merge_2
z
rnn_2/while/Switch_3Switchrnn_2/while/Merge_3rnn_2/while/LoopCond*
T0*&
_class
loc:@rnn_2/while/Merge_3
z
rnn_2/while/Switch_4Switchrnn_2/while/Merge_4rnn_2/while/LoopCond*
T0*&
_class
loc:@rnn_2/while/Merge_4
?
rnn_2/while/IdentityIdentityrnn_2/while/Switch:1*
T0
C
rnn_2/while/Identity_1Identityrnn_2/while/Switch_1:1*
T0
C
rnn_2/while/Identity_2Identityrnn_2/while/Switch_2:1*
T0
C
rnn_2/while/Identity_3Identityrnn_2/while/Switch_3:1*
T0
C
rnn_2/while/Identity_4Identityrnn_2/while/Switch_4:1*
T0
R
rnn_2/while/add/yConst^rnn_2/while/Identity*
value	B :*
dtype0
H
rnn_2/while/addAddrnn_2/while/Identityrnn_2/while/add/y*
T0
�
rnn_2/while/TensorArrayReadV3TensorArrayReadV3#rnn_2/while/TensorArrayReadV3/Enterrnn_2/while/Identity_1%rnn_2/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_2/while/TensorArrayReadV3/EnterEnterrnn_2/TensorArray_1*
parallel_iterations *)

frame_namernn_2/while/while_context*
T0*
is_constant(
�
%rnn_2/while/TensorArrayReadV3/Enter_1Enter@rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
2rnn/LSTM_2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@rnn/LSTM_2/kernel*
valueB"�   �  *
dtype0
�
0rnn/LSTM_2/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@rnn/LSTM_2/kernel*
valueB
 *����
�
0rnn/LSTM_2/kernel/Initializer/random_uniform/maxConst*
dtype0*$
_class
loc:@rnn/LSTM_2/kernel*
valueB
 *���=
�
:rnn/LSTM_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_2/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
dtype0*
seed2�*
seed�
�
0rnn/LSTM_2/kernel/Initializer/random_uniform/subSub0rnn/LSTM_2/kernel/Initializer/random_uniform/max0rnn/LSTM_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_2/kernel
�
0rnn/LSTM_2/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_2/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_2/kernel
�
,rnn/LSTM_2/kernel/Initializer/random_uniformAdd0rnn/LSTM_2/kernel/Initializer/random_uniform/mul0rnn/LSTM_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_2/kernel
�
rnn/LSTM_2/kernel
VariableV2*$
_class
loc:@rnn/LSTM_2/kernel*
dtype0*
	container *
shape:
��*
shared_name 
�
rnn/LSTM_2/kernel/AssignAssignrnn/LSTM_2/kernel,rnn/LSTM_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_2/kernel
>
rnn/LSTM_2/kernel/readIdentityrnn/LSTM_2/kernel*
T0
w
!rnn/LSTM_2/bias/Initializer/zerosConst*"
_class
loc:@rnn/LSTM_2/bias*
valueB�*    *
dtype0
�
rnn/LSTM_2/bias
VariableV2*
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_2/bias*
dtype0*
	container 
�
rnn/LSTM_2/bias/AssignAssignrnn/LSTM_2/bias!rnn/LSTM_2/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_2/bias*
validate_shape(
:
rnn/LSTM_2/bias/readIdentityrnn/LSTM_2/bias*
T0
_
rnn_2/while/LSTM_2/concat/axisConst^rnn_2/while/Identity*
dtype0*
value	B :
�
rnn_2/while/LSTM_2/concatConcatV2rnn_2/while/TensorArrayReadV3rnn_2/while/Identity_4rnn_2/while/LSTM_2/concat/axis*

Tidx0*
T0*
N
�
rnn_2/while/LSTM_2/MatMulMatMulrnn_2/while/LSTM_2/concatrnn_2/while/LSTM_2/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
�
rnn_2/while/LSTM_2/MatMul/EnterEnterrnn/LSTM_2/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
rnn_2/while/LSTM_2/BiasAddBiasAddrnn_2/while/LSTM_2/MatMul rnn_2/while/LSTM_2/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn_2/while/LSTM_2/BiasAdd/EnterEnterrnn/LSTM_2/bias/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
Y
rnn_2/while/LSTM_2/ConstConst^rnn_2/while/Identity*
value	B :*
dtype0
c
"rnn_2/while/LSTM_2/split/split_dimConst^rnn_2/while/Identity*
value	B :*
dtype0
{
rnn_2/while/LSTM_2/splitSplit"rnn_2/while/LSTM_2/split/split_dimrnn_2/while/LSTM_2/BiasAdd*
T0*
	num_split
\
rnn_2/while/LSTM_2/add/yConst^rnn_2/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_2/while/LSTM_2/addAddrnn_2/while/LSTM_2/split:2rnn_2/while/LSTM_2/add/y*
T0
F
rnn_2/while/LSTM_2/SigmoidSigmoidrnn_2/while/LSTM_2/add*
T0
Z
rnn_2/while/LSTM_2/mulMulrnn_2/while/LSTM_2/Sigmoidrnn_2/while/Identity_3*
T0
J
rnn_2/while/LSTM_2/Sigmoid_1Sigmoidrnn_2/while/LSTM_2/split*
T0
D
rnn_2/while/LSTM_2/TanhTanhrnn_2/while/LSTM_2/split:1*
T0
_
rnn_2/while/LSTM_2/mul_1Mulrnn_2/while/LSTM_2/Sigmoid_1rnn_2/while/LSTM_2/Tanh*
T0
Z
rnn_2/while/LSTM_2/add_1Addrnn_2/while/LSTM_2/mulrnn_2/while/LSTM_2/mul_1*
T0
L
rnn_2/while/LSTM_2/Sigmoid_2Sigmoidrnn_2/while/LSTM_2/split:3*
T0
D
rnn_2/while/LSTM_2/Tanh_1Tanhrnn_2/while/LSTM_2/add_1*
T0
a
rnn_2/while/LSTM_2/mul_2Mulrnn_2/while/LSTM_2/Sigmoid_2rnn_2/while/LSTM_2/Tanh_1*
T0
�
/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_2/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_2/while/Identity_1rnn_2/while/LSTM_2/mul_2rnn_2/while/Identity_2*
T0*+
_class!
loc:@rnn_2/while/LSTM_2/mul_2
�
5rnn_2/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_2/TensorArray*
parallel_iterations *)

frame_namernn_2/while/while_context*
T0*+
_class!
loc:@rnn_2/while/LSTM_2/mul_2*
is_constant(
T
rnn_2/while/add_1/yConst^rnn_2/while/Identity*
dtype0*
value	B :
N
rnn_2/while/add_1Addrnn_2/while/Identity_1rnn_2/while/add_1/y*
T0
D
rnn_2/while/NextIterationNextIterationrnn_2/while/add*
T0
H
rnn_2/while/NextIteration_1NextIterationrnn_2/while/add_1*
T0
f
rnn_2/while/NextIteration_2NextIteration/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_2/while/NextIteration_3NextIterationrnn_2/while/LSTM_2/add_1*
T0
O
rnn_2/while/NextIteration_4NextIterationrnn_2/while/LSTM_2/mul_2*
T0
5
rnn_2/while/ExitExitrnn_2/while/Switch*
T0
9
rnn_2/while/Exit_1Exitrnn_2/while/Switch_1*
T0
9
rnn_2/while/Exit_2Exitrnn_2/while/Switch_2*
T0
9
rnn_2/while/Exit_3Exitrnn_2/while/Switch_3*
T0
9
rnn_2/while/Exit_4Exitrnn_2/while/Switch_4*
T0
�
(rnn_2/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_2/TensorArrayrnn_2/while/Exit_2*$
_class
loc:@rnn_2/TensorArray
r
"rnn_2/TensorArrayStack/range/startConst*$
_class
loc:@rnn_2/TensorArray*
value	B : *
dtype0
r
"rnn_2/TensorArrayStack/range/deltaConst*$
_class
loc:@rnn_2/TensorArray*
value	B :*
dtype0
�
rnn_2/TensorArrayStack/rangeRange"rnn_2/TensorArrayStack/range/start(rnn_2/TensorArrayStack/TensorArraySizeV3"rnn_2/TensorArrayStack/range/delta*

Tidx0*$
_class
loc:@rnn_2/TensorArray
�
*rnn_2/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_2/TensorArrayrnn_2/TensorArrayStack/rangernn_2/while/Exit_2*
dtype0*$
element_shape:���������x*$
_class
loc:@rnn_2/TensorArray
;
rnn_2/Const_1Const*
dtype0*
valueB:x
6
rnn_2/Rank_1Const*
value	B :*
dtype0
=
rnn_2/range_1/startConst*
value	B :*
dtype0
=
rnn_2/range_1/deltaConst*
dtype0*
value	B :
Z
rnn_2/range_1Rangernn_2/range_1/startrnn_2/Rank_1rnn_2/range_1/delta*

Tidx0
L
rnn_2/concat_2/values_0Const*
valueB"       *
dtype0
=
rnn_2/concat_2/axisConst*
dtype0*
value	B : 
u
rnn_2/concat_2ConcatV2rnn_2/concat_2/values_0rnn_2/range_1rnn_2/concat_2/axis*

Tidx0*
T0*
N
p
rnn_2/transpose_1	Transpose*rnn_2/TensorArrayStack/TensorArrayGatherV3rnn_2/concat_2*
Tperm0*
T0
4

rnn_3/RankConst*
value	B :*
dtype0
;
rnn_3/range/startConst*
value	B :*
dtype0
;
rnn_3/range/deltaConst*
dtype0*
value	B :
R
rnn_3/rangeRangernn_3/range/start
rnn_3/Rankrnn_3/range/delta*

Tidx0
J
rnn_3/concat/values_0Const*
valueB"       *
dtype0
;
rnn_3/concat/axisConst*
dtype0*
value	B : 
m
rnn_3/concatConcatV2rnn_3/concat/values_0rnn_3/rangernn_3/concat/axis*
T0*
N*

Tidx0
S
rnn_3/transpose	Transposernn_2/transpose_1rnn_3/concat*
T0*
Tperm0
>
rnn_3/ShapeShapernn_3/transpose*
T0*
out_type0
G
rnn_3/strided_slice/stackConst*
dtype0*
valueB:
I
rnn_3/strided_slice/stack_1Const*
valueB:*
dtype0
I
rnn_3/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_3/strided_sliceStridedSlicernn_3/Shapernn_3/strided_slice/stackrnn_3/strided_slice/stack_1rnn_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
P
&rnn_3/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
"rnn_3/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_3/strided_slice&rnn_3/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
K
rnn_3/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
M
#rnn_3/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
rnn_3/LSTMCellZeroState/concatConcatV2"rnn_3/LSTMCellZeroState/ExpandDimsrnn_3/LSTMCellZeroState/Const#rnn_3/LSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
P
#rnn_3/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
rnn_3/LSTMCellZeroState/zerosFillrnn_3/LSTMCellZeroState/concat#rnn_3/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_3/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : 
�
$rnn_3/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_3/strided_slice(rnn_3/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0
M
rnn_3/LSTMCellZeroState/Const_1Const*
valueB:x*
dtype0
R
(rnn_3/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
$rnn_3/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_3/strided_slice(rnn_3/LSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0
M
rnn_3/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
O
%rnn_3/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
 rnn_3/LSTMCellZeroState/concat_1ConcatV2$rnn_3/LSTMCellZeroState/ExpandDims_2rnn_3/LSTMCellZeroState/Const_2%rnn_3/LSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0
R
%rnn_3/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn_3/LSTMCellZeroState/zeros_1Fill rnn_3/LSTMCellZeroState/concat_1%rnn_3/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_3/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
$rnn_3/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_3/strided_slice(rnn_3/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
M
rnn_3/LSTMCellZeroState/Const_3Const*
valueB:x*
dtype0
@
rnn_3/Shape_1Shapernn_3/transpose*
T0*
out_type0
I
rnn_3/strided_slice_1/stackConst*
valueB: *
dtype0
K
rnn_3/strided_slice_1/stack_1Const*
valueB:*
dtype0
K
rnn_3/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
rnn_3/strided_slice_1StridedSlicernn_3/Shape_1rnn_3/strided_slice_1/stackrnn_3/strided_slice_1/stack_1rnn_3/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
@
rnn_3/Shape_2Shapernn_3/transpose*
T0*
out_type0
I
rnn_3/strided_slice_2/stackConst*
dtype0*
valueB:
K
rnn_3/strided_slice_2/stack_1Const*
dtype0*
valueB:
K
rnn_3/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
rnn_3/strided_slice_2StridedSlicernn_3/Shape_2rnn_3/strided_slice_2/stackrnn_3/strided_slice_2/stack_1rnn_3/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
>
rnn_3/ExpandDims/dimConst*
value	B : *
dtype0
`
rnn_3/ExpandDims
ExpandDimsrnn_3/strided_slice_2rnn_3/ExpandDims/dim*

Tdim0*
T0
9
rnn_3/ConstConst*
valueB:x*
dtype0
=
rnn_3/concat_1/axisConst*
dtype0*
value	B : 
l
rnn_3/concat_1ConcatV2rnn_3/ExpandDimsrnn_3/Constrnn_3/concat_1/axis*
N*

Tidx0*
T0
>
rnn_3/zeros/ConstConst*
valueB
 *    *
dtype0
Q
rnn_3/zerosFillrnn_3/concat_1rnn_3/zeros/Const*
T0*

index_type0
4

rnn_3/timeConst*
value	B : *
dtype0
�
rnn_3/TensorArrayTensorArrayV3rnn_3/strided_slice_1*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*1
tensor_array_namernn_3/dynamic_rnn/output_0
�
rnn_3/TensorArray_1TensorArrayV3rnn_3/strided_slice_1*
dtype0*$
element_shape:���������x*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*0
tensor_array_namernn_3/dynamic_rnn/input_0
Q
rnn_3/TensorArrayUnstack/ShapeShapernn_3/transpose*
T0*
out_type0
Z
,rnn_3/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
\
.rnn_3/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
\
.rnn_3/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
�
&rnn_3/TensorArrayUnstack/strided_sliceStridedSlicernn_3/TensorArrayUnstack/Shape,rnn_3/TensorArrayUnstack/strided_slice/stack.rnn_3/TensorArrayUnstack/strided_slice/stack_1.rnn_3/TensorArrayUnstack/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
N
$rnn_3/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
N
$rnn_3/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn_3/TensorArrayUnstack/rangeRange$rnn_3/TensorArrayUnstack/range/start&rnn_3/TensorArrayUnstack/strided_slice$rnn_3/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_3/TensorArray_1rnn_3/TensorArrayUnstack/rangernn_3/transposernn_3/TensorArray_1:1*
T0*"
_class
loc:@rnn_3/transpose
9
rnn_3/Maximum/xConst*
value	B :*
dtype0
I
rnn_3/MaximumMaximumrnn_3/Maximum/xrnn_3/strided_slice_1*
T0
G
rnn_3/MinimumMinimumrnn_3/strided_slice_1rnn_3/Maximum*
T0
G
rnn_3/while/iteration_counterConst*
value	B : *
dtype0
�
rnn_3/while/EnterEnterrnn_3/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_3/while/while_context
�
rnn_3/while/Enter_1Enter
rnn_3/time*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant( 
�
rnn_3/while/Enter_2Enterrnn_3/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_3/while/while_context
�
rnn_3/while/Enter_3Enterrnn_3/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_3/while/while_context
�
rnn_3/while/Enter_4Enterrnn_3/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_3/while/while_context
Z
rnn_3/while/MergeMergernn_3/while/Enterrnn_3/while/NextIteration*
T0*
N
`
rnn_3/while/Merge_1Mergernn_3/while/Enter_1rnn_3/while/NextIteration_1*
T0*
N
`
rnn_3/while/Merge_2Mergernn_3/while/Enter_2rnn_3/while/NextIteration_2*
N*
T0
`
rnn_3/while/Merge_3Mergernn_3/while/Enter_3rnn_3/while/NextIteration_3*
T0*
N
`
rnn_3/while/Merge_4Mergernn_3/while/Enter_4rnn_3/while/NextIteration_4*
T0*
N
L
rnn_3/while/LessLessrnn_3/while/Mergernn_3/while/Less/Enter*
T0
�
rnn_3/while/Less/EnterEnterrnn_3/strided_slice_1*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant(
R
rnn_3/while/Less_1Lessrnn_3/while/Merge_1rnn_3/while/Less_1/Enter*
T0
�
rnn_3/while/Less_1/EnterEnterrnn_3/Minimum*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
J
rnn_3/while/LogicalAnd
LogicalAndrnn_3/while/Lessrnn_3/while/Less_1
8
rnn_3/while/LoopCondLoopCondrnn_3/while/LogicalAnd
t
rnn_3/while/SwitchSwitchrnn_3/while/Mergernn_3/while/LoopCond*
T0*$
_class
loc:@rnn_3/while/Merge
z
rnn_3/while/Switch_1Switchrnn_3/while/Merge_1rnn_3/while/LoopCond*
T0*&
_class
loc:@rnn_3/while/Merge_1
z
rnn_3/while/Switch_2Switchrnn_3/while/Merge_2rnn_3/while/LoopCond*
T0*&
_class
loc:@rnn_3/while/Merge_2
z
rnn_3/while/Switch_3Switchrnn_3/while/Merge_3rnn_3/while/LoopCond*
T0*&
_class
loc:@rnn_3/while/Merge_3
z
rnn_3/while/Switch_4Switchrnn_3/while/Merge_4rnn_3/while/LoopCond*
T0*&
_class
loc:@rnn_3/while/Merge_4
?
rnn_3/while/IdentityIdentityrnn_3/while/Switch:1*
T0
C
rnn_3/while/Identity_1Identityrnn_3/while/Switch_1:1*
T0
C
rnn_3/while/Identity_2Identityrnn_3/while/Switch_2:1*
T0
C
rnn_3/while/Identity_3Identityrnn_3/while/Switch_3:1*
T0
C
rnn_3/while/Identity_4Identityrnn_3/while/Switch_4:1*
T0
R
rnn_3/while/add/yConst^rnn_3/while/Identity*
value	B :*
dtype0
H
rnn_3/while/addAddrnn_3/while/Identityrnn_3/while/add/y*
T0
�
rnn_3/while/TensorArrayReadV3TensorArrayReadV3#rnn_3/while/TensorArrayReadV3/Enterrnn_3/while/Identity_1%rnn_3/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_3/while/TensorArrayReadV3/EnterEnterrnn_3/TensorArray_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
%rnn_3/while/TensorArrayReadV3/Enter_1Enter@rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
2rnn/LSTM_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@rnn/LSTM_3/kernel*
valueB"�   �  
�
0rnn/LSTM_3/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@rnn/LSTM_3/kernel*
valueB
 *����
�
0rnn/LSTM_3/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@rnn/LSTM_3/kernel*
valueB
 *���=*
dtype0
�
:rnn/LSTM_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_3/kernel/Initializer/random_uniform/shape*
dtype0*
seed2�*
seed�*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
0rnn/LSTM_3/kernel/Initializer/random_uniform/subSub0rnn/LSTM_3/kernel/Initializer/random_uniform/max0rnn/LSTM_3/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
0rnn/LSTM_3/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_3/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_3/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
,rnn/LSTM_3/kernel/Initializer/random_uniformAdd0rnn/LSTM_3/kernel/Initializer/random_uniform/mul0rnn/LSTM_3/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
rnn/LSTM_3/kernel
VariableV2*
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_3/kernel*
dtype0*
	container 
�
rnn/LSTM_3/kernel/AssignAssignrnn/LSTM_3/kernel,rnn/LSTM_3/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_3/kernel*
validate_shape(
>
rnn/LSTM_3/kernel/readIdentityrnn/LSTM_3/kernel*
T0
w
!rnn/LSTM_3/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@rnn/LSTM_3/bias*
valueB�*    
�
rnn/LSTM_3/bias
VariableV2*"
_class
loc:@rnn/LSTM_3/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_3/bias/AssignAssignrnn/LSTM_3/bias!rnn/LSTM_3/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_3/bias*
validate_shape(
:
rnn/LSTM_3/bias/readIdentityrnn/LSTM_3/bias*
T0
_
rnn_3/while/LSTM_3/concat/axisConst^rnn_3/while/Identity*
value	B :*
dtype0
�
rnn_3/while/LSTM_3/concatConcatV2rnn_3/while/TensorArrayReadV3rnn_3/while/Identity_4rnn_3/while/LSTM_3/concat/axis*
T0*
N*

Tidx0
�
rnn_3/while/LSTM_3/MatMulMatMulrnn_3/while/LSTM_3/concatrnn_3/while/LSTM_3/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
rnn_3/while/LSTM_3/MatMul/EnterEnterrnn/LSTM_3/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
rnn_3/while/LSTM_3/BiasAddBiasAddrnn_3/while/LSTM_3/MatMul rnn_3/while/LSTM_3/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn_3/while/LSTM_3/BiasAdd/EnterEnterrnn/LSTM_3/bias/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
Y
rnn_3/while/LSTM_3/ConstConst^rnn_3/while/Identity*
value	B :*
dtype0
c
"rnn_3/while/LSTM_3/split/split_dimConst^rnn_3/while/Identity*
value	B :*
dtype0
{
rnn_3/while/LSTM_3/splitSplit"rnn_3/while/LSTM_3/split/split_dimrnn_3/while/LSTM_3/BiasAdd*
T0*
	num_split
\
rnn_3/while/LSTM_3/add/yConst^rnn_3/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_3/while/LSTM_3/addAddrnn_3/while/LSTM_3/split:2rnn_3/while/LSTM_3/add/y*
T0
F
rnn_3/while/LSTM_3/SigmoidSigmoidrnn_3/while/LSTM_3/add*
T0
Z
rnn_3/while/LSTM_3/mulMulrnn_3/while/LSTM_3/Sigmoidrnn_3/while/Identity_3*
T0
J
rnn_3/while/LSTM_3/Sigmoid_1Sigmoidrnn_3/while/LSTM_3/split*
T0
D
rnn_3/while/LSTM_3/TanhTanhrnn_3/while/LSTM_3/split:1*
T0
_
rnn_3/while/LSTM_3/mul_1Mulrnn_3/while/LSTM_3/Sigmoid_1rnn_3/while/LSTM_3/Tanh*
T0
Z
rnn_3/while/LSTM_3/add_1Addrnn_3/while/LSTM_3/mulrnn_3/while/LSTM_3/mul_1*
T0
L
rnn_3/while/LSTM_3/Sigmoid_2Sigmoidrnn_3/while/LSTM_3/split:3*
T0
D
rnn_3/while/LSTM_3/Tanh_1Tanhrnn_3/while/LSTM_3/add_1*
T0
a
rnn_3/while/LSTM_3/mul_2Mulrnn_3/while/LSTM_3/Sigmoid_2rnn_3/while/LSTM_3/Tanh_1*
T0
�
/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_3/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_3/while/Identity_1rnn_3/while/LSTM_3/mul_2rnn_3/while/Identity_2*
T0*+
_class!
loc:@rnn_3/while/LSTM_3/mul_2
�
5rnn_3/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_3/TensorArray*
T0*+
_class!
loc:@rnn_3/while/LSTM_3/mul_2*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
T
rnn_3/while/add_1/yConst^rnn_3/while/Identity*
value	B :*
dtype0
N
rnn_3/while/add_1Addrnn_3/while/Identity_1rnn_3/while/add_1/y*
T0
D
rnn_3/while/NextIterationNextIterationrnn_3/while/add*
T0
H
rnn_3/while/NextIteration_1NextIterationrnn_3/while/add_1*
T0
f
rnn_3/while/NextIteration_2NextIteration/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_3/while/NextIteration_3NextIterationrnn_3/while/LSTM_3/add_1*
T0
O
rnn_3/while/NextIteration_4NextIterationrnn_3/while/LSTM_3/mul_2*
T0
5
rnn_3/while/ExitExitrnn_3/while/Switch*
T0
9
rnn_3/while/Exit_1Exitrnn_3/while/Switch_1*
T0
9
rnn_3/while/Exit_2Exitrnn_3/while/Switch_2*
T0
9
rnn_3/while/Exit_3Exitrnn_3/while/Switch_3*
T0
9
rnn_3/while/Exit_4Exitrnn_3/while/Switch_4*
T0
�
(rnn_3/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_3/TensorArrayrnn_3/while/Exit_2*$
_class
loc:@rnn_3/TensorArray
r
"rnn_3/TensorArrayStack/range/startConst*
dtype0*$
_class
loc:@rnn_3/TensorArray*
value	B : 
r
"rnn_3/TensorArrayStack/range/deltaConst*$
_class
loc:@rnn_3/TensorArray*
value	B :*
dtype0
�
rnn_3/TensorArrayStack/rangeRange"rnn_3/TensorArrayStack/range/start(rnn_3/TensorArrayStack/TensorArraySizeV3"rnn_3/TensorArrayStack/range/delta*

Tidx0*$
_class
loc:@rnn_3/TensorArray
�
*rnn_3/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_3/TensorArrayrnn_3/TensorArrayStack/rangernn_3/while/Exit_2*$
element_shape:���������x*$
_class
loc:@rnn_3/TensorArray*
dtype0
;
rnn_3/Const_1Const*
valueB:x*
dtype0
6
rnn_3/Rank_1Const*
value	B :*
dtype0
=
rnn_3/range_1/startConst*
value	B :*
dtype0
=
rnn_3/range_1/deltaConst*
value	B :*
dtype0
Z
rnn_3/range_1Rangernn_3/range_1/startrnn_3/Rank_1rnn_3/range_1/delta*

Tidx0
L
rnn_3/concat_2/values_0Const*
dtype0*
valueB"       
=
rnn_3/concat_2/axisConst*
dtype0*
value	B : 
u
rnn_3/concat_2ConcatV2rnn_3/concat_2/values_0rnn_3/range_1rnn_3/concat_2/axis*
T0*
N*

Tidx0
p
rnn_3/transpose_1	Transpose*rnn_3/TensorArrayStack/TensorArrayGatherV3rnn_3/concat_2*
Tperm0*
T0
4

rnn_4/RankConst*
dtype0*
value	B :
;
rnn_4/range/startConst*
dtype0*
value	B :
;
rnn_4/range/deltaConst*
value	B :*
dtype0
R
rnn_4/rangeRangernn_4/range/start
rnn_4/Rankrnn_4/range/delta*

Tidx0
J
rnn_4/concat/values_0Const*
valueB"       *
dtype0
;
rnn_4/concat/axisConst*
value	B : *
dtype0
m
rnn_4/concatConcatV2rnn_4/concat/values_0rnn_4/rangernn_4/concat/axis*
N*

Tidx0*
T0
S
rnn_4/transpose	Transposernn_3/transpose_1rnn_4/concat*
Tperm0*
T0
>
rnn_4/ShapeShapernn_4/transpose*
T0*
out_type0
G
rnn_4/strided_slice/stackConst*
valueB:*
dtype0
I
rnn_4/strided_slice/stack_1Const*
valueB:*
dtype0
I
rnn_4/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_4/strided_sliceStridedSlicernn_4/Shapernn_4/strided_slice/stackrnn_4/strided_slice/stack_1rnn_4/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
P
&rnn_4/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
�
"rnn_4/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_4/strided_slice&rnn_4/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
K
rnn_4/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
M
#rnn_4/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0
�
rnn_4/LSTMCellZeroState/concatConcatV2"rnn_4/LSTMCellZeroState/ExpandDimsrnn_4/LSTMCellZeroState/Const#rnn_4/LSTMCellZeroState/concat/axis*
N*

Tidx0*
T0
P
#rnn_4/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
rnn_4/LSTMCellZeroState/zerosFillrnn_4/LSTMCellZeroState/concat#rnn_4/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_4/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0
�
$rnn_4/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_4/strided_slice(rnn_4/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0
M
rnn_4/LSTMCellZeroState/Const_1Const*
dtype0*
valueB:x
R
(rnn_4/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
$rnn_4/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_4/strided_slice(rnn_4/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
M
rnn_4/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
O
%rnn_4/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
 rnn_4/LSTMCellZeroState/concat_1ConcatV2$rnn_4/LSTMCellZeroState/ExpandDims_2rnn_4/LSTMCellZeroState/Const_2%rnn_4/LSTMCellZeroState/concat_1/axis*
N*

Tidx0*
T0
R
%rnn_4/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn_4/LSTMCellZeroState/zeros_1Fill rnn_4/LSTMCellZeroState/concat_1%rnn_4/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_4/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : 
�
$rnn_4/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_4/strided_slice(rnn_4/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
M
rnn_4/LSTMCellZeroState/Const_3Const*
valueB:x*
dtype0
@
rnn_4/Shape_1Shapernn_4/transpose*
T0*
out_type0
I
rnn_4/strided_slice_1/stackConst*
valueB: *
dtype0
K
rnn_4/strided_slice_1/stack_1Const*
dtype0*
valueB:
K
rnn_4/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
rnn_4/strided_slice_1StridedSlicernn_4/Shape_1rnn_4/strided_slice_1/stackrnn_4/strided_slice_1/stack_1rnn_4/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
@
rnn_4/Shape_2Shapernn_4/transpose*
T0*
out_type0
I
rnn_4/strided_slice_2/stackConst*
valueB:*
dtype0
K
rnn_4/strided_slice_2/stack_1Const*
valueB:*
dtype0
K
rnn_4/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
rnn_4/strided_slice_2StridedSlicernn_4/Shape_2rnn_4/strided_slice_2/stackrnn_4/strided_slice_2/stack_1rnn_4/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
>
rnn_4/ExpandDims/dimConst*
value	B : *
dtype0
`
rnn_4/ExpandDims
ExpandDimsrnn_4/strided_slice_2rnn_4/ExpandDims/dim*

Tdim0*
T0
9
rnn_4/ConstConst*
valueB:x*
dtype0
=
rnn_4/concat_1/axisConst*
value	B : *
dtype0
l
rnn_4/concat_1ConcatV2rnn_4/ExpandDimsrnn_4/Constrnn_4/concat_1/axis*

Tidx0*
T0*
N
>
rnn_4/zeros/ConstConst*
dtype0*
valueB
 *    
Q
rnn_4/zerosFillrnn_4/concat_1rnn_4/zeros/Const*
T0*

index_type0
4

rnn_4/timeConst*
value	B : *
dtype0
�
rnn_4/TensorArrayTensorArrayV3rnn_4/strided_slice_1*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*1
tensor_array_namernn_4/dynamic_rnn/output_0
�
rnn_4/TensorArray_1TensorArrayV3rnn_4/strided_slice_1*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*0
tensor_array_namernn_4/dynamic_rnn/input_0
Q
rnn_4/TensorArrayUnstack/ShapeShapernn_4/transpose*
T0*
out_type0
Z
,rnn_4/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
\
.rnn_4/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
\
.rnn_4/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
�
&rnn_4/TensorArrayUnstack/strided_sliceStridedSlicernn_4/TensorArrayUnstack/Shape,rnn_4/TensorArrayUnstack/strided_slice/stack.rnn_4/TensorArrayUnstack/strided_slice/stack_1.rnn_4/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
N
$rnn_4/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
N
$rnn_4/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn_4/TensorArrayUnstack/rangeRange$rnn_4/TensorArrayUnstack/range/start&rnn_4/TensorArrayUnstack/strided_slice$rnn_4/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_4/TensorArray_1rnn_4/TensorArrayUnstack/rangernn_4/transposernn_4/TensorArray_1:1*
T0*"
_class
loc:@rnn_4/transpose
9
rnn_4/Maximum/xConst*
value	B :*
dtype0
I
rnn_4/MaximumMaximumrnn_4/Maximum/xrnn_4/strided_slice_1*
T0
G
rnn_4/MinimumMinimumrnn_4/strided_slice_1rnn_4/Maximum*
T0
G
rnn_4/while/iteration_counterConst*
value	B : *
dtype0
�
rnn_4/while/EnterEnterrnn_4/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_4/while/while_context
�
rnn_4/while/Enter_1Enter
rnn_4/time*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant( 
�
rnn_4/while/Enter_2Enterrnn_4/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_4/while/while_context
�
rnn_4/while/Enter_3Enterrnn_4/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_4/while/while_context
�
rnn_4/while/Enter_4Enterrnn_4/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_4/while/while_context
Z
rnn_4/while/MergeMergernn_4/while/Enterrnn_4/while/NextIteration*
N*
T0
`
rnn_4/while/Merge_1Mergernn_4/while/Enter_1rnn_4/while/NextIteration_1*
T0*
N
`
rnn_4/while/Merge_2Mergernn_4/while/Enter_2rnn_4/while/NextIteration_2*
T0*
N
`
rnn_4/while/Merge_3Mergernn_4/while/Enter_3rnn_4/while/NextIteration_3*
N*
T0
`
rnn_4/while/Merge_4Mergernn_4/while/Enter_4rnn_4/while/NextIteration_4*
N*
T0
L
rnn_4/while/LessLessrnn_4/while/Mergernn_4/while/Less/Enter*
T0
�
rnn_4/while/Less/EnterEnterrnn_4/strided_slice_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
R
rnn_4/while/Less_1Lessrnn_4/while/Merge_1rnn_4/while/Less_1/Enter*
T0
�
rnn_4/while/Less_1/EnterEnterrnn_4/Minimum*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
J
rnn_4/while/LogicalAnd
LogicalAndrnn_4/while/Lessrnn_4/while/Less_1
8
rnn_4/while/LoopCondLoopCondrnn_4/while/LogicalAnd
t
rnn_4/while/SwitchSwitchrnn_4/while/Mergernn_4/while/LoopCond*
T0*$
_class
loc:@rnn_4/while/Merge
z
rnn_4/while/Switch_1Switchrnn_4/while/Merge_1rnn_4/while/LoopCond*
T0*&
_class
loc:@rnn_4/while/Merge_1
z
rnn_4/while/Switch_2Switchrnn_4/while/Merge_2rnn_4/while/LoopCond*
T0*&
_class
loc:@rnn_4/while/Merge_2
z
rnn_4/while/Switch_3Switchrnn_4/while/Merge_3rnn_4/while/LoopCond*
T0*&
_class
loc:@rnn_4/while/Merge_3
z
rnn_4/while/Switch_4Switchrnn_4/while/Merge_4rnn_4/while/LoopCond*
T0*&
_class
loc:@rnn_4/while/Merge_4
?
rnn_4/while/IdentityIdentityrnn_4/while/Switch:1*
T0
C
rnn_4/while/Identity_1Identityrnn_4/while/Switch_1:1*
T0
C
rnn_4/while/Identity_2Identityrnn_4/while/Switch_2:1*
T0
C
rnn_4/while/Identity_3Identityrnn_4/while/Switch_3:1*
T0
C
rnn_4/while/Identity_4Identityrnn_4/while/Switch_4:1*
T0
R
rnn_4/while/add/yConst^rnn_4/while/Identity*
value	B :*
dtype0
H
rnn_4/while/addAddrnn_4/while/Identityrnn_4/while/add/y*
T0
�
rnn_4/while/TensorArrayReadV3TensorArrayReadV3#rnn_4/while/TensorArrayReadV3/Enterrnn_4/while/Identity_1%rnn_4/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_4/while/TensorArrayReadV3/EnterEnterrnn_4/TensorArray_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
%rnn_4/while/TensorArrayReadV3/Enter_1Enter@rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
2rnn/LSTM_4/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@rnn/LSTM_4/kernel*
valueB"�   �  *
dtype0
�
0rnn/LSTM_4/kernel/Initializer/random_uniform/minConst*$
_class
loc:@rnn/LSTM_4/kernel*
valueB
 *����*
dtype0
�
0rnn/LSTM_4/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@rnn/LSTM_4/kernel*
valueB
 *���=*
dtype0
�
:rnn/LSTM_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_4/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
dtype0*
seed2�*
seed�
�
0rnn/LSTM_4/kernel/Initializer/random_uniform/subSub0rnn/LSTM_4/kernel/Initializer/random_uniform/max0rnn/LSTM_4/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
0rnn/LSTM_4/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_4/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_4/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
,rnn/LSTM_4/kernel/Initializer/random_uniformAdd0rnn/LSTM_4/kernel/Initializer/random_uniform/mul0rnn/LSTM_4/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
rnn/LSTM_4/kernel
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_4/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_4/kernel/AssignAssignrnn/LSTM_4/kernel,rnn/LSTM_4/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
validate_shape(
>
rnn/LSTM_4/kernel/readIdentityrnn/LSTM_4/kernel*
T0
w
!rnn/LSTM_4/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@rnn/LSTM_4/bias*
valueB�*    
�
rnn/LSTM_4/bias
VariableV2*
shared_name *"
_class
loc:@rnn/LSTM_4/bias*
dtype0*
	container *
shape:�
�
rnn/LSTM_4/bias/AssignAssignrnn/LSTM_4/bias!rnn/LSTM_4/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_4/bias
:
rnn/LSTM_4/bias/readIdentityrnn/LSTM_4/bias*
T0
_
rnn_4/while/LSTM_4/concat/axisConst^rnn_4/while/Identity*
dtype0*
value	B :
�
rnn_4/while/LSTM_4/concatConcatV2rnn_4/while/TensorArrayReadV3rnn_4/while/Identity_4rnn_4/while/LSTM_4/concat/axis*
T0*
N*

Tidx0
�
rnn_4/while/LSTM_4/MatMulMatMulrnn_4/while/LSTM_4/concatrnn_4/while/LSTM_4/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
rnn_4/while/LSTM_4/MatMul/EnterEnterrnn/LSTM_4/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
rnn_4/while/LSTM_4/BiasAddBiasAddrnn_4/while/LSTM_4/MatMul rnn_4/while/LSTM_4/BiasAdd/Enter*
data_formatNHWC*
T0
�
 rnn_4/while/LSTM_4/BiasAdd/EnterEnterrnn/LSTM_4/bias/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
Y
rnn_4/while/LSTM_4/ConstConst^rnn_4/while/Identity*
value	B :*
dtype0
c
"rnn_4/while/LSTM_4/split/split_dimConst^rnn_4/while/Identity*
value	B :*
dtype0
{
rnn_4/while/LSTM_4/splitSplit"rnn_4/while/LSTM_4/split/split_dimrnn_4/while/LSTM_4/BiasAdd*
T0*
	num_split
\
rnn_4/while/LSTM_4/add/yConst^rnn_4/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_4/while/LSTM_4/addAddrnn_4/while/LSTM_4/split:2rnn_4/while/LSTM_4/add/y*
T0
F
rnn_4/while/LSTM_4/SigmoidSigmoidrnn_4/while/LSTM_4/add*
T0
Z
rnn_4/while/LSTM_4/mulMulrnn_4/while/LSTM_4/Sigmoidrnn_4/while/Identity_3*
T0
J
rnn_4/while/LSTM_4/Sigmoid_1Sigmoidrnn_4/while/LSTM_4/split*
T0
D
rnn_4/while/LSTM_4/TanhTanhrnn_4/while/LSTM_4/split:1*
T0
_
rnn_4/while/LSTM_4/mul_1Mulrnn_4/while/LSTM_4/Sigmoid_1rnn_4/while/LSTM_4/Tanh*
T0
Z
rnn_4/while/LSTM_4/add_1Addrnn_4/while/LSTM_4/mulrnn_4/while/LSTM_4/mul_1*
T0
L
rnn_4/while/LSTM_4/Sigmoid_2Sigmoidrnn_4/while/LSTM_4/split:3*
T0
D
rnn_4/while/LSTM_4/Tanh_1Tanhrnn_4/while/LSTM_4/add_1*
T0
a
rnn_4/while/LSTM_4/mul_2Mulrnn_4/while/LSTM_4/Sigmoid_2rnn_4/while/LSTM_4/Tanh_1*
T0
�
/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_4/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_4/while/Identity_1rnn_4/while/LSTM_4/mul_2rnn_4/while/Identity_2*
T0*+
_class!
loc:@rnn_4/while/LSTM_4/mul_2
�
5rnn_4/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_4/TensorArray*
T0*+
_class!
loc:@rnn_4/while/LSTM_4/mul_2*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
T
rnn_4/while/add_1/yConst^rnn_4/while/Identity*
value	B :*
dtype0
N
rnn_4/while/add_1Addrnn_4/while/Identity_1rnn_4/while/add_1/y*
T0
D
rnn_4/while/NextIterationNextIterationrnn_4/while/add*
T0
H
rnn_4/while/NextIteration_1NextIterationrnn_4/while/add_1*
T0
f
rnn_4/while/NextIteration_2NextIteration/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_4/while/NextIteration_3NextIterationrnn_4/while/LSTM_4/add_1*
T0
O
rnn_4/while/NextIteration_4NextIterationrnn_4/while/LSTM_4/mul_2*
T0
5
rnn_4/while/ExitExitrnn_4/while/Switch*
T0
9
rnn_4/while/Exit_1Exitrnn_4/while/Switch_1*
T0
9
rnn_4/while/Exit_2Exitrnn_4/while/Switch_2*
T0
9
rnn_4/while/Exit_3Exitrnn_4/while/Switch_3*
T0
9
rnn_4/while/Exit_4Exitrnn_4/while/Switch_4*
T0
�
(rnn_4/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_4/TensorArrayrnn_4/while/Exit_2*$
_class
loc:@rnn_4/TensorArray
r
"rnn_4/TensorArrayStack/range/startConst*$
_class
loc:@rnn_4/TensorArray*
value	B : *
dtype0
r
"rnn_4/TensorArrayStack/range/deltaConst*$
_class
loc:@rnn_4/TensorArray*
value	B :*
dtype0
�
rnn_4/TensorArrayStack/rangeRange"rnn_4/TensorArrayStack/range/start(rnn_4/TensorArrayStack/TensorArraySizeV3"rnn_4/TensorArrayStack/range/delta*

Tidx0*$
_class
loc:@rnn_4/TensorArray
�
*rnn_4/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_4/TensorArrayrnn_4/TensorArrayStack/rangernn_4/while/Exit_2*
dtype0*$
element_shape:���������x*$
_class
loc:@rnn_4/TensorArray
;
rnn_4/Const_1Const*
valueB:x*
dtype0
6
rnn_4/Rank_1Const*
value	B :*
dtype0
=
rnn_4/range_1/startConst*
value	B :*
dtype0
=
rnn_4/range_1/deltaConst*
value	B :*
dtype0
Z
rnn_4/range_1Rangernn_4/range_1/startrnn_4/Rank_1rnn_4/range_1/delta*

Tidx0
L
rnn_4/concat_2/values_0Const*
valueB"       *
dtype0
=
rnn_4/concat_2/axisConst*
dtype0*
value	B : 
u
rnn_4/concat_2ConcatV2rnn_4/concat_2/values_0rnn_4/range_1rnn_4/concat_2/axis*
T0*
N*

Tidx0
p
rnn_4/transpose_1	Transpose*rnn_4/TensorArrayStack/TensorArrayGatherV3rnn_4/concat_2*
Tperm0*
T0
4

rnn_5/RankConst*
value	B :*
dtype0
;
rnn_5/range/startConst*
value	B :*
dtype0
;
rnn_5/range/deltaConst*
dtype0*
value	B :
R
rnn_5/rangeRangernn_5/range/start
rnn_5/Rankrnn_5/range/delta*

Tidx0
J
rnn_5/concat/values_0Const*
valueB"       *
dtype0
;
rnn_5/concat/axisConst*
value	B : *
dtype0
m
rnn_5/concatConcatV2rnn_5/concat/values_0rnn_5/rangernn_5/concat/axis*
T0*
N*

Tidx0
S
rnn_5/transpose	Transposernn_4/transpose_1rnn_5/concat*
Tperm0*
T0
>
rnn_5/ShapeShapernn_5/transpose*
T0*
out_type0
G
rnn_5/strided_slice/stackConst*
dtype0*
valueB:
I
rnn_5/strided_slice/stack_1Const*
valueB:*
dtype0
I
rnn_5/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_5/strided_sliceStridedSlicernn_5/Shapernn_5/strided_slice/stackrnn_5/strided_slice/stack_1rnn_5/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
P
&rnn_5/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
�
"rnn_5/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_5/strided_slice&rnn_5/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
K
rnn_5/LSTMCellZeroState/ConstConst*
valueB:x*
dtype0
M
#rnn_5/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
rnn_5/LSTMCellZeroState/concatConcatV2"rnn_5/LSTMCellZeroState/ExpandDimsrnn_5/LSTMCellZeroState/Const#rnn_5/LSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
P
#rnn_5/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
rnn_5/LSTMCellZeroState/zerosFillrnn_5/LSTMCellZeroState/concat#rnn_5/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_5/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0
�
$rnn_5/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_5/strided_slice(rnn_5/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0
M
rnn_5/LSTMCellZeroState/Const_1Const*
valueB:x*
dtype0
R
(rnn_5/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
$rnn_5/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_5/strided_slice(rnn_5/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
M
rnn_5/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
O
%rnn_5/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
 rnn_5/LSTMCellZeroState/concat_1ConcatV2$rnn_5/LSTMCellZeroState/ExpandDims_2rnn_5/LSTMCellZeroState/Const_2%rnn_5/LSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0
R
%rnn_5/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn_5/LSTMCellZeroState/zeros_1Fill rnn_5/LSTMCellZeroState/concat_1%rnn_5/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_5/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : 
�
$rnn_5/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_5/strided_slice(rnn_5/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0
M
rnn_5/LSTMCellZeroState/Const_3Const*
valueB:x*
dtype0
@
rnn_5/Shape_1Shapernn_5/transpose*
T0*
out_type0
I
rnn_5/strided_slice_1/stackConst*
dtype0*
valueB: 
K
rnn_5/strided_slice_1/stack_1Const*
valueB:*
dtype0
K
rnn_5/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
rnn_5/strided_slice_1StridedSlicernn_5/Shape_1rnn_5/strided_slice_1/stackrnn_5/strided_slice_1/stack_1rnn_5/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
@
rnn_5/Shape_2Shapernn_5/transpose*
T0*
out_type0
I
rnn_5/strided_slice_2/stackConst*
dtype0*
valueB:
K
rnn_5/strided_slice_2/stack_1Const*
valueB:*
dtype0
K
rnn_5/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
rnn_5/strided_slice_2StridedSlicernn_5/Shape_2rnn_5/strided_slice_2/stackrnn_5/strided_slice_2/stack_1rnn_5/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
>
rnn_5/ExpandDims/dimConst*
value	B : *
dtype0
`
rnn_5/ExpandDims
ExpandDimsrnn_5/strided_slice_2rnn_5/ExpandDims/dim*

Tdim0*
T0
9
rnn_5/ConstConst*
valueB:x*
dtype0
=
rnn_5/concat_1/axisConst*
dtype0*
value	B : 
l
rnn_5/concat_1ConcatV2rnn_5/ExpandDimsrnn_5/Constrnn_5/concat_1/axis*

Tidx0*
T0*
N
>
rnn_5/zeros/ConstConst*
dtype0*
valueB
 *    
Q
rnn_5/zerosFillrnn_5/concat_1rnn_5/zeros/Const*
T0*

index_type0
4

rnn_5/timeConst*
dtype0*
value	B : 
�
rnn_5/TensorArrayTensorArrayV3rnn_5/strided_slice_1*$
element_shape:���������x*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*1
tensor_array_namernn_5/dynamic_rnn/output_0*
dtype0
�
rnn_5/TensorArray_1TensorArrayV3rnn_5/strided_slice_1*
identical_element_shapes(*0
tensor_array_namernn_5/dynamic_rnn/input_0*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(
Q
rnn_5/TensorArrayUnstack/ShapeShapernn_5/transpose*
T0*
out_type0
Z
,rnn_5/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
\
.rnn_5/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
\
.rnn_5/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
&rnn_5/TensorArrayUnstack/strided_sliceStridedSlicernn_5/TensorArrayUnstack/Shape,rnn_5/TensorArrayUnstack/strided_slice/stack.rnn_5/TensorArrayUnstack/strided_slice/stack_1.rnn_5/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
N
$rnn_5/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
N
$rnn_5/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn_5/TensorArrayUnstack/rangeRange$rnn_5/TensorArrayUnstack/range/start&rnn_5/TensorArrayUnstack/strided_slice$rnn_5/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_5/TensorArray_1rnn_5/TensorArrayUnstack/rangernn_5/transposernn_5/TensorArray_1:1*
T0*"
_class
loc:@rnn_5/transpose
9
rnn_5/Maximum/xConst*
value	B :*
dtype0
I
rnn_5/MaximumMaximumrnn_5/Maximum/xrnn_5/strided_slice_1*
T0
G
rnn_5/MinimumMinimumrnn_5/strided_slice_1rnn_5/Maximum*
T0
G
rnn_5/while/iteration_counterConst*
dtype0*
value	B : 
�
rnn_5/while/EnterEnterrnn_5/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_5/while/while_context
�
rnn_5/while/Enter_1Enter
rnn_5/time*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_5/while/while_context
�
rnn_5/while/Enter_2Enterrnn_5/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_5/while/while_context
�
rnn_5/while/Enter_3Enterrnn_5/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_5/while/while_context
�
rnn_5/while/Enter_4Enterrnn_5/LSTMCellZeroState/zeros_1*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant( 
Z
rnn_5/while/MergeMergernn_5/while/Enterrnn_5/while/NextIteration*
T0*
N
`
rnn_5/while/Merge_1Mergernn_5/while/Enter_1rnn_5/while/NextIteration_1*
T0*
N
`
rnn_5/while/Merge_2Mergernn_5/while/Enter_2rnn_5/while/NextIteration_2*
T0*
N
`
rnn_5/while/Merge_3Mergernn_5/while/Enter_3rnn_5/while/NextIteration_3*
T0*
N
`
rnn_5/while/Merge_4Mergernn_5/while/Enter_4rnn_5/while/NextIteration_4*
N*
T0
L
rnn_5/while/LessLessrnn_5/while/Mergernn_5/while/Less/Enter*
T0
�
rnn_5/while/Less/EnterEnterrnn_5/strided_slice_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
R
rnn_5/while/Less_1Lessrnn_5/while/Merge_1rnn_5/while/Less_1/Enter*
T0
�
rnn_5/while/Less_1/EnterEnterrnn_5/Minimum*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
J
rnn_5/while/LogicalAnd
LogicalAndrnn_5/while/Lessrnn_5/while/Less_1
8
rnn_5/while/LoopCondLoopCondrnn_5/while/LogicalAnd
t
rnn_5/while/SwitchSwitchrnn_5/while/Mergernn_5/while/LoopCond*
T0*$
_class
loc:@rnn_5/while/Merge
z
rnn_5/while/Switch_1Switchrnn_5/while/Merge_1rnn_5/while/LoopCond*
T0*&
_class
loc:@rnn_5/while/Merge_1
z
rnn_5/while/Switch_2Switchrnn_5/while/Merge_2rnn_5/while/LoopCond*
T0*&
_class
loc:@rnn_5/while/Merge_2
z
rnn_5/while/Switch_3Switchrnn_5/while/Merge_3rnn_5/while/LoopCond*
T0*&
_class
loc:@rnn_5/while/Merge_3
z
rnn_5/while/Switch_4Switchrnn_5/while/Merge_4rnn_5/while/LoopCond*
T0*&
_class
loc:@rnn_5/while/Merge_4
?
rnn_5/while/IdentityIdentityrnn_5/while/Switch:1*
T0
C
rnn_5/while/Identity_1Identityrnn_5/while/Switch_1:1*
T0
C
rnn_5/while/Identity_2Identityrnn_5/while/Switch_2:1*
T0
C
rnn_5/while/Identity_3Identityrnn_5/while/Switch_3:1*
T0
C
rnn_5/while/Identity_4Identityrnn_5/while/Switch_4:1*
T0
R
rnn_5/while/add/yConst^rnn_5/while/Identity*
value	B :*
dtype0
H
rnn_5/while/addAddrnn_5/while/Identityrnn_5/while/add/y*
T0
�
rnn_5/while/TensorArrayReadV3TensorArrayReadV3#rnn_5/while/TensorArrayReadV3/Enterrnn_5/while/Identity_1%rnn_5/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_5/while/TensorArrayReadV3/EnterEnterrnn_5/TensorArray_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
%rnn_5/while/TensorArrayReadV3/Enter_1Enter@rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
2rnn/LSTM_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@rnn/LSTM_5/kernel*
valueB"�   �  
�
0rnn/LSTM_5/kernel/Initializer/random_uniform/minConst*$
_class
loc:@rnn/LSTM_5/kernel*
valueB
 *����*
dtype0
�
0rnn/LSTM_5/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@rnn/LSTM_5/kernel*
valueB
 *���=*
dtype0
�
:rnn/LSTM_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_5/kernel/Initializer/random_uniform/shape*
dtype0*
seed2�*
seed�*
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
0rnn/LSTM_5/kernel/Initializer/random_uniform/subSub0rnn/LSTM_5/kernel/Initializer/random_uniform/max0rnn/LSTM_5/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
0rnn/LSTM_5/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_5/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_5/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
,rnn/LSTM_5/kernel/Initializer/random_uniformAdd0rnn/LSTM_5/kernel/Initializer/random_uniform/mul0rnn/LSTM_5/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
rnn/LSTM_5/kernel
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_5/kernel
�
rnn/LSTM_5/kernel/AssignAssignrnn/LSTM_5/kernel,rnn/LSTM_5/kernel/Initializer/random_uniform*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(*
use_locking(
>
rnn/LSTM_5/kernel/readIdentityrnn/LSTM_5/kernel*
T0
w
!rnn/LSTM_5/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@rnn/LSTM_5/bias*
valueB�*    
�
rnn/LSTM_5/bias
VariableV2*"
_class
loc:@rnn/LSTM_5/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_5/bias/AssignAssignrnn/LSTM_5/bias!rnn/LSTM_5/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias
:
rnn/LSTM_5/bias/readIdentityrnn/LSTM_5/bias*
T0
_
rnn_5/while/LSTM_5/concat/axisConst^rnn_5/while/Identity*
value	B :*
dtype0
�
rnn_5/while/LSTM_5/concatConcatV2rnn_5/while/TensorArrayReadV3rnn_5/while/Identity_4rnn_5/while/LSTM_5/concat/axis*

Tidx0*
T0*
N
�
rnn_5/while/LSTM_5/MatMulMatMulrnn_5/while/LSTM_5/concatrnn_5/while/LSTM_5/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
�
rnn_5/while/LSTM_5/MatMul/EnterEnterrnn/LSTM_5/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
rnn_5/while/LSTM_5/BiasAddBiasAddrnn_5/while/LSTM_5/MatMul rnn_5/while/LSTM_5/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn_5/while/LSTM_5/BiasAdd/EnterEnterrnn/LSTM_5/bias/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
Y
rnn_5/while/LSTM_5/ConstConst^rnn_5/while/Identity*
value	B :*
dtype0
c
"rnn_5/while/LSTM_5/split/split_dimConst^rnn_5/while/Identity*
value	B :*
dtype0
{
rnn_5/while/LSTM_5/splitSplit"rnn_5/while/LSTM_5/split/split_dimrnn_5/while/LSTM_5/BiasAdd*
T0*
	num_split
\
rnn_5/while/LSTM_5/add/yConst^rnn_5/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_5/while/LSTM_5/addAddrnn_5/while/LSTM_5/split:2rnn_5/while/LSTM_5/add/y*
T0
F
rnn_5/while/LSTM_5/SigmoidSigmoidrnn_5/while/LSTM_5/add*
T0
Z
rnn_5/while/LSTM_5/mulMulrnn_5/while/LSTM_5/Sigmoidrnn_5/while/Identity_3*
T0
J
rnn_5/while/LSTM_5/Sigmoid_1Sigmoidrnn_5/while/LSTM_5/split*
T0
D
rnn_5/while/LSTM_5/TanhTanhrnn_5/while/LSTM_5/split:1*
T0
_
rnn_5/while/LSTM_5/mul_1Mulrnn_5/while/LSTM_5/Sigmoid_1rnn_5/while/LSTM_5/Tanh*
T0
Z
rnn_5/while/LSTM_5/add_1Addrnn_5/while/LSTM_5/mulrnn_5/while/LSTM_5/mul_1*
T0
L
rnn_5/while/LSTM_5/Sigmoid_2Sigmoidrnn_5/while/LSTM_5/split:3*
T0
D
rnn_5/while/LSTM_5/Tanh_1Tanhrnn_5/while/LSTM_5/add_1*
T0
a
rnn_5/while/LSTM_5/mul_2Mulrnn_5/while/LSTM_5/Sigmoid_2rnn_5/while/LSTM_5/Tanh_1*
T0
�
/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_5/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_5/while/Identity_1rnn_5/while/LSTM_5/mul_2rnn_5/while/Identity_2*
T0*+
_class!
loc:@rnn_5/while/LSTM_5/mul_2
�
5rnn_5/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_5/TensorArray*
T0*+
_class!
loc:@rnn_5/while/LSTM_5/mul_2*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
T
rnn_5/while/add_1/yConst^rnn_5/while/Identity*
value	B :*
dtype0
N
rnn_5/while/add_1Addrnn_5/while/Identity_1rnn_5/while/add_1/y*
T0
D
rnn_5/while/NextIterationNextIterationrnn_5/while/add*
T0
H
rnn_5/while/NextIteration_1NextIterationrnn_5/while/add_1*
T0
f
rnn_5/while/NextIteration_2NextIteration/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_5/while/NextIteration_3NextIterationrnn_5/while/LSTM_5/add_1*
T0
O
rnn_5/while/NextIteration_4NextIterationrnn_5/while/LSTM_5/mul_2*
T0
5
rnn_5/while/ExitExitrnn_5/while/Switch*
T0
9
rnn_5/while/Exit_1Exitrnn_5/while/Switch_1*
T0
9
rnn_5/while/Exit_2Exitrnn_5/while/Switch_2*
T0
9
rnn_5/while/Exit_3Exitrnn_5/while/Switch_3*
T0
9
rnn_5/while/Exit_4Exitrnn_5/while/Switch_4*
T0
�
(rnn_5/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_5/TensorArrayrnn_5/while/Exit_2*$
_class
loc:@rnn_5/TensorArray
r
"rnn_5/TensorArrayStack/range/startConst*$
_class
loc:@rnn_5/TensorArray*
value	B : *
dtype0
r
"rnn_5/TensorArrayStack/range/deltaConst*$
_class
loc:@rnn_5/TensorArray*
value	B :*
dtype0
�
rnn_5/TensorArrayStack/rangeRange"rnn_5/TensorArrayStack/range/start(rnn_5/TensorArrayStack/TensorArraySizeV3"rnn_5/TensorArrayStack/range/delta*

Tidx0*$
_class
loc:@rnn_5/TensorArray
�
*rnn_5/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_5/TensorArrayrnn_5/TensorArrayStack/rangernn_5/while/Exit_2*$
element_shape:���������x*$
_class
loc:@rnn_5/TensorArray*
dtype0
;
rnn_5/Const_1Const*
valueB:x*
dtype0
6
rnn_5/Rank_1Const*
value	B :*
dtype0
=
rnn_5/range_1/startConst*
value	B :*
dtype0
=
rnn_5/range_1/deltaConst*
value	B :*
dtype0
Z
rnn_5/range_1Rangernn_5/range_1/startrnn_5/Rank_1rnn_5/range_1/delta*

Tidx0
L
rnn_5/concat_2/values_0Const*
valueB"       *
dtype0
=
rnn_5/concat_2/axisConst*
dtype0*
value	B : 
u
rnn_5/concat_2ConcatV2rnn_5/concat_2/values_0rnn_5/range_1rnn_5/concat_2/axis*
T0*
N*

Tidx0
p
rnn_5/transpose_1	Transpose*rnn_5/TensorArrayStack/TensorArrayGatherV3rnn_5/concat_2*
T0*
Tperm0
4

rnn_6/RankConst*
value	B :*
dtype0
;
rnn_6/range/startConst*
dtype0*
value	B :
;
rnn_6/range/deltaConst*
value	B :*
dtype0
R
rnn_6/rangeRangernn_6/range/start
rnn_6/Rankrnn_6/range/delta*

Tidx0
J
rnn_6/concat/values_0Const*
dtype0*
valueB"       
;
rnn_6/concat/axisConst*
dtype0*
value	B : 
m
rnn_6/concatConcatV2rnn_6/concat/values_0rnn_6/rangernn_6/concat/axis*
T0*
N*

Tidx0
S
rnn_6/transpose	Transposernn_5/transpose_1rnn_6/concat*
Tperm0*
T0
>
rnn_6/ShapeShapernn_6/transpose*
T0*
out_type0
G
rnn_6/strided_slice/stackConst*
dtype0*
valueB:
I
rnn_6/strided_slice/stack_1Const*
valueB:*
dtype0
I
rnn_6/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn_6/strided_sliceStridedSlicernn_6/Shapernn_6/strided_slice/stackrnn_6/strided_slice/stack_1rnn_6/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
P
&rnn_6/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
"rnn_6/LSTMCellZeroState/ExpandDims
ExpandDimsrnn_6/strided_slice&rnn_6/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
K
rnn_6/LSTMCellZeroState/ConstConst*
dtype0*
valueB:x
M
#rnn_6/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0
�
rnn_6/LSTMCellZeroState/concatConcatV2"rnn_6/LSTMCellZeroState/ExpandDimsrnn_6/LSTMCellZeroState/Const#rnn_6/LSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
P
#rnn_6/LSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    
�
rnn_6/LSTMCellZeroState/zerosFillrnn_6/LSTMCellZeroState/concat#rnn_6/LSTMCellZeroState/zeros/Const*
T0*

index_type0
R
(rnn_6/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0
�
$rnn_6/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn_6/strided_slice(rnn_6/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0
M
rnn_6/LSTMCellZeroState/Const_1Const*
valueB:x*
dtype0
R
(rnn_6/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
$rnn_6/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn_6/strided_slice(rnn_6/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
M
rnn_6/LSTMCellZeroState/Const_2Const*
valueB:x*
dtype0
O
%rnn_6/LSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : 
�
 rnn_6/LSTMCellZeroState/concat_1ConcatV2$rnn_6/LSTMCellZeroState/ExpandDims_2rnn_6/LSTMCellZeroState/Const_2%rnn_6/LSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0
R
%rnn_6/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
rnn_6/LSTMCellZeroState/zeros_1Fill rnn_6/LSTMCellZeroState/concat_1%rnn_6/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
R
(rnn_6/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
$rnn_6/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn_6/strided_slice(rnn_6/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
M
rnn_6/LSTMCellZeroState/Const_3Const*
valueB:x*
dtype0
@
rnn_6/Shape_1Shapernn_6/transpose*
T0*
out_type0
I
rnn_6/strided_slice_1/stackConst*
dtype0*
valueB: 
K
rnn_6/strided_slice_1/stack_1Const*
valueB:*
dtype0
K
rnn_6/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
rnn_6/strided_slice_1StridedSlicernn_6/Shape_1rnn_6/strided_slice_1/stackrnn_6/strided_slice_1/stack_1rnn_6/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
@
rnn_6/Shape_2Shapernn_6/transpose*
T0*
out_type0
I
rnn_6/strided_slice_2/stackConst*
valueB:*
dtype0
K
rnn_6/strided_slice_2/stack_1Const*
valueB:*
dtype0
K
rnn_6/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
rnn_6/strided_slice_2StridedSlicernn_6/Shape_2rnn_6/strided_slice_2/stackrnn_6/strided_slice_2/stack_1rnn_6/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
>
rnn_6/ExpandDims/dimConst*
value	B : *
dtype0
`
rnn_6/ExpandDims
ExpandDimsrnn_6/strided_slice_2rnn_6/ExpandDims/dim*

Tdim0*
T0
9
rnn_6/ConstConst*
valueB:x*
dtype0
=
rnn_6/concat_1/axisConst*
dtype0*
value	B : 
l
rnn_6/concat_1ConcatV2rnn_6/ExpandDimsrnn_6/Constrnn_6/concat_1/axis*
N*

Tidx0*
T0
>
rnn_6/zeros/ConstConst*
valueB
 *    *
dtype0
Q
rnn_6/zerosFillrnn_6/concat_1rnn_6/zeros/Const*
T0*

index_type0
4

rnn_6/timeConst*
value	B : *
dtype0
�
rnn_6/TensorArrayTensorArrayV3rnn_6/strided_slice_1*1
tensor_array_namernn_6/dynamic_rnn/output_0*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
rnn_6/TensorArray_1TensorArrayV3rnn_6/strided_slice_1*0
tensor_array_namernn_6/dynamic_rnn/input_0*
dtype0*$
element_shape:���������x*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
Q
rnn_6/TensorArrayUnstack/ShapeShapernn_6/transpose*
T0*
out_type0
Z
,rnn_6/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
\
.rnn_6/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
\
.rnn_6/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
&rnn_6/TensorArrayUnstack/strided_sliceStridedSlicernn_6/TensorArrayUnstack/Shape,rnn_6/TensorArrayUnstack/strided_slice/stack.rnn_6/TensorArrayUnstack/strided_slice/stack_1.rnn_6/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
N
$rnn_6/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
N
$rnn_6/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn_6/TensorArrayUnstack/rangeRange$rnn_6/TensorArrayUnstack/range/start&rnn_6/TensorArrayUnstack/strided_slice$rnn_6/TensorArrayUnstack/range/delta*

Tidx0
�
@rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_6/TensorArray_1rnn_6/TensorArrayUnstack/rangernn_6/transposernn_6/TensorArray_1:1*
T0*"
_class
loc:@rnn_6/transpose
9
rnn_6/Maximum/xConst*
value	B :*
dtype0
I
rnn_6/MaximumMaximumrnn_6/Maximum/xrnn_6/strided_slice_1*
T0
G
rnn_6/MinimumMinimumrnn_6/strided_slice_1rnn_6/Maximum*
T0
G
rnn_6/while/iteration_counterConst*
value	B : *
dtype0
�
rnn_6/while/EnterEnterrnn_6/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
�
rnn_6/while/Enter_1Enter
rnn_6/time*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
�
rnn_6/while/Enter_2Enterrnn_6/TensorArray:1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
�
rnn_6/while/Enter_3Enterrnn_6/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
�
rnn_6/while/Enter_4Enterrnn_6/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
Z
rnn_6/while/MergeMergernn_6/while/Enterrnn_6/while/NextIteration*
T0*
N
`
rnn_6/while/Merge_1Mergernn_6/while/Enter_1rnn_6/while/NextIteration_1*
N*
T0
`
rnn_6/while/Merge_2Mergernn_6/while/Enter_2rnn_6/while/NextIteration_2*
T0*
N
`
rnn_6/while/Merge_3Mergernn_6/while/Enter_3rnn_6/while/NextIteration_3*
T0*
N
`
rnn_6/while/Merge_4Mergernn_6/while/Enter_4rnn_6/while/NextIteration_4*
N*
T0
L
rnn_6/while/LessLessrnn_6/while/Mergernn_6/while/Less/Enter*
T0
�
rnn_6/while/Less/EnterEnterrnn_6/strided_slice_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
R
rnn_6/while/Less_1Lessrnn_6/while/Merge_1rnn_6/while/Less_1/Enter*
T0
�
rnn_6/while/Less_1/EnterEnterrnn_6/Minimum*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
J
rnn_6/while/LogicalAnd
LogicalAndrnn_6/while/Lessrnn_6/while/Less_1
8
rnn_6/while/LoopCondLoopCondrnn_6/while/LogicalAnd
t
rnn_6/while/SwitchSwitchrnn_6/while/Mergernn_6/while/LoopCond*
T0*$
_class
loc:@rnn_6/while/Merge
z
rnn_6/while/Switch_1Switchrnn_6/while/Merge_1rnn_6/while/LoopCond*
T0*&
_class
loc:@rnn_6/while/Merge_1
z
rnn_6/while/Switch_2Switchrnn_6/while/Merge_2rnn_6/while/LoopCond*
T0*&
_class
loc:@rnn_6/while/Merge_2
z
rnn_6/while/Switch_3Switchrnn_6/while/Merge_3rnn_6/while/LoopCond*
T0*&
_class
loc:@rnn_6/while/Merge_3
z
rnn_6/while/Switch_4Switchrnn_6/while/Merge_4rnn_6/while/LoopCond*
T0*&
_class
loc:@rnn_6/while/Merge_4
?
rnn_6/while/IdentityIdentityrnn_6/while/Switch:1*
T0
C
rnn_6/while/Identity_1Identityrnn_6/while/Switch_1:1*
T0
C
rnn_6/while/Identity_2Identityrnn_6/while/Switch_2:1*
T0
C
rnn_6/while/Identity_3Identityrnn_6/while/Switch_3:1*
T0
C
rnn_6/while/Identity_4Identityrnn_6/while/Switch_4:1*
T0
R
rnn_6/while/add/yConst^rnn_6/while/Identity*
dtype0*
value	B :
H
rnn_6/while/addAddrnn_6/while/Identityrnn_6/while/add/y*
T0
�
rnn_6/while/TensorArrayReadV3TensorArrayReadV3#rnn_6/while/TensorArrayReadV3/Enterrnn_6/while/Identity_1%rnn_6/while/TensorArrayReadV3/Enter_1*
dtype0
�
#rnn_6/while/TensorArrayReadV3/EnterEnterrnn_6/TensorArray_1*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
�
%rnn_6/while/TensorArrayReadV3/Enter_1Enter@rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
2rnn/LSTM_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@rnn/LSTM_6/kernel*
valueB"�   �  
�
0rnn/LSTM_6/kernel/Initializer/random_uniform/minConst*$
_class
loc:@rnn/LSTM_6/kernel*
valueB
 *����*
dtype0
�
0rnn/LSTM_6/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@rnn/LSTM_6/kernel*
valueB
 *���=*
dtype0
�
:rnn/LSTM_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform2rnn/LSTM_6/kernel/Initializer/random_uniform/shape*
seed�*
T0*$
_class
loc:@rnn/LSTM_6/kernel*
dtype0*
seed2�
�
0rnn/LSTM_6/kernel/Initializer/random_uniform/subSub0rnn/LSTM_6/kernel/Initializer/random_uniform/max0rnn/LSTM_6/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
0rnn/LSTM_6/kernel/Initializer/random_uniform/mulMul:rnn/LSTM_6/kernel/Initializer/random_uniform/RandomUniform0rnn/LSTM_6/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
,rnn/LSTM_6/kernel/Initializer/random_uniformAdd0rnn/LSTM_6/kernel/Initializer/random_uniform/mul0rnn/LSTM_6/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
rnn/LSTM_6/kernel
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_6/kernel
�
rnn/LSTM_6/kernel/AssignAssignrnn/LSTM_6/kernel,rnn/LSTM_6/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_6/kernel
>
rnn/LSTM_6/kernel/readIdentityrnn/LSTM_6/kernel*
T0
w
!rnn/LSTM_6/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@rnn/LSTM_6/bias*
valueB�*    
�
rnn/LSTM_6/bias
VariableV2*
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_6/bias*
dtype0*
	container 
�
rnn/LSTM_6/bias/AssignAssignrnn/LSTM_6/bias!rnn/LSTM_6/bias/Initializer/zeros*
T0*"
_class
loc:@rnn/LSTM_6/bias*
validate_shape(*
use_locking(
:
rnn/LSTM_6/bias/readIdentityrnn/LSTM_6/bias*
T0
_
rnn_6/while/LSTM_6/concat/axisConst^rnn_6/while/Identity*
value	B :*
dtype0
�
rnn_6/while/LSTM_6/concatConcatV2rnn_6/while/TensorArrayReadV3rnn_6/while/Identity_4rnn_6/while/LSTM_6/concat/axis*
T0*
N*

Tidx0
�
rnn_6/while/LSTM_6/MatMulMatMulrnn_6/while/LSTM_6/concatrnn_6/while/LSTM_6/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
rnn_6/while/LSTM_6/MatMul/EnterEnterrnn/LSTM_6/kernel/read*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
rnn_6/while/LSTM_6/BiasAddBiasAddrnn_6/while/LSTM_6/MatMul rnn_6/while/LSTM_6/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn_6/while/LSTM_6/BiasAdd/EnterEnterrnn/LSTM_6/bias/read*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
Y
rnn_6/while/LSTM_6/ConstConst^rnn_6/while/Identity*
value	B :*
dtype0
c
"rnn_6/while/LSTM_6/split/split_dimConst^rnn_6/while/Identity*
value	B :*
dtype0
{
rnn_6/while/LSTM_6/splitSplit"rnn_6/while/LSTM_6/split/split_dimrnn_6/while/LSTM_6/BiasAdd*
	num_split*
T0
\
rnn_6/while/LSTM_6/add/yConst^rnn_6/while/Identity*
valueB
 *  �?*
dtype0
\
rnn_6/while/LSTM_6/addAddrnn_6/while/LSTM_6/split:2rnn_6/while/LSTM_6/add/y*
T0
F
rnn_6/while/LSTM_6/SigmoidSigmoidrnn_6/while/LSTM_6/add*
T0
Z
rnn_6/while/LSTM_6/mulMulrnn_6/while/LSTM_6/Sigmoidrnn_6/while/Identity_3*
T0
J
rnn_6/while/LSTM_6/Sigmoid_1Sigmoidrnn_6/while/LSTM_6/split*
T0
D
rnn_6/while/LSTM_6/TanhTanhrnn_6/while/LSTM_6/split:1*
T0
_
rnn_6/while/LSTM_6/mul_1Mulrnn_6/while/LSTM_6/Sigmoid_1rnn_6/while/LSTM_6/Tanh*
T0
Z
rnn_6/while/LSTM_6/add_1Addrnn_6/while/LSTM_6/mulrnn_6/while/LSTM_6/mul_1*
T0
L
rnn_6/while/LSTM_6/Sigmoid_2Sigmoidrnn_6/while/LSTM_6/split:3*
T0
D
rnn_6/while/LSTM_6/Tanh_1Tanhrnn_6/while/LSTM_6/add_1*
T0
a
rnn_6/while/LSTM_6/mul_2Mulrnn_6/while/LSTM_6/Sigmoid_2rnn_6/while/LSTM_6/Tanh_1*
T0
�
/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_6/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_6/while/Identity_1rnn_6/while/LSTM_6/mul_2rnn_6/while/Identity_2*
T0*+
_class!
loc:@rnn_6/while/LSTM_6/mul_2
�
5rnn_6/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_6/TensorArray*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*+
_class!
loc:@rnn_6/while/LSTM_6/mul_2*
is_constant(
T
rnn_6/while/add_1/yConst^rnn_6/while/Identity*
value	B :*
dtype0
N
rnn_6/while/add_1Addrnn_6/while/Identity_1rnn_6/while/add_1/y*
T0
D
rnn_6/while/NextIterationNextIterationrnn_6/while/add*
T0
H
rnn_6/while/NextIteration_1NextIterationrnn_6/while/add_1*
T0
f
rnn_6/while/NextIteration_2NextIteration/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3*
T0
O
rnn_6/while/NextIteration_3NextIterationrnn_6/while/LSTM_6/add_1*
T0
O
rnn_6/while/NextIteration_4NextIterationrnn_6/while/LSTM_6/mul_2*
T0
5
rnn_6/while/ExitExitrnn_6/while/Switch*
T0
9
rnn_6/while/Exit_1Exitrnn_6/while/Switch_1*
T0
9
rnn_6/while/Exit_2Exitrnn_6/while/Switch_2*
T0
9
rnn_6/while/Exit_3Exitrnn_6/while/Switch_3*
T0
9
rnn_6/while/Exit_4Exitrnn_6/while/Switch_4*
T0
�
(rnn_6/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_6/TensorArrayrnn_6/while/Exit_2*$
_class
loc:@rnn_6/TensorArray
r
"rnn_6/TensorArrayStack/range/startConst*$
_class
loc:@rnn_6/TensorArray*
value	B : *
dtype0
r
"rnn_6/TensorArrayStack/range/deltaConst*
dtype0*$
_class
loc:@rnn_6/TensorArray*
value	B :
�
rnn_6/TensorArrayStack/rangeRange"rnn_6/TensorArrayStack/range/start(rnn_6/TensorArrayStack/TensorArraySizeV3"rnn_6/TensorArrayStack/range/delta*

Tidx0*$
_class
loc:@rnn_6/TensorArray
�
*rnn_6/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_6/TensorArrayrnn_6/TensorArrayStack/rangernn_6/while/Exit_2*$
element_shape:���������x*$
_class
loc:@rnn_6/TensorArray*
dtype0
;
rnn_6/Const_1Const*
valueB:x*
dtype0
6
rnn_6/Rank_1Const*
dtype0*
value	B :
=
rnn_6/range_1/startConst*
value	B :*
dtype0
=
rnn_6/range_1/deltaConst*
value	B :*
dtype0
Z
rnn_6/range_1Rangernn_6/range_1/startrnn_6/Rank_1rnn_6/range_1/delta*

Tidx0
L
rnn_6/concat_2/values_0Const*
valueB"       *
dtype0
=
rnn_6/concat_2/axisConst*
value	B : *
dtype0
u
rnn_6/concat_2ConcatV2rnn_6/concat_2/values_0rnn_6/range_1rnn_6/concat_2/axis*
T0*
N*

Tidx0
p
rnn_6/transpose_1	Transpose*rnn_6/TensorArrayStack/TensorArrayGatherV3rnn_6/concat_2*
T0*
Tperm0
H
strided_slice/stackConst*
valueB"    ����*
dtype0
J
strided_slice/stack_1Const*
valueB"        *
dtype0
J
strided_slice/stack_2Const*
valueB"      *
dtype0
�
strided_sliceStridedSlicernn_6/transpose_1strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
�
8fully_connected/weights/Initializer/random_uniform/shapeConst**
_class 
loc:@fully_connected/weights*
valueB"x      *
dtype0
�
6fully_connected/weights/Initializer/random_uniform/minConst**
_class 
loc:@fully_connected/weights*
valueB
 *�t_�*
dtype0
�
6fully_connected/weights/Initializer/random_uniform/maxConst**
_class 
loc:@fully_connected/weights*
valueB
 *�t_>*
dtype0
�
@fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniform8fully_connected/weights/Initializer/random_uniform/shape*
T0**
_class 
loc:@fully_connected/weights*
dtype0*
seed2�*
seed�
�
6fully_connected/weights/Initializer/random_uniform/subSub6fully_connected/weights/Initializer/random_uniform/max6fully_connected/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@fully_connected/weights
�
6fully_connected/weights/Initializer/random_uniform/mulMul@fully_connected/weights/Initializer/random_uniform/RandomUniform6fully_connected/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@fully_connected/weights
�
2fully_connected/weights/Initializer/random_uniformAdd6fully_connected/weights/Initializer/random_uniform/mul6fully_connected/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@fully_connected/weights
�
fully_connected/weights
VariableV2*
shared_name **
_class 
loc:@fully_connected/weights*
dtype0*
	container *
shape
:x
�
fully_connected/weights/AssignAssignfully_connected/weights2fully_connected/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@fully_connected/weights
v
fully_connected/weights/readIdentityfully_connected/weights*
T0**
_class 
loc:@fully_connected/weights
�
(fully_connected/biases/Initializer/zerosConst*)
_class
loc:@fully_connected/biases*
valueB*    *
dtype0
�
fully_connected/biases
VariableV2*
dtype0*
	container *
shape:*
shared_name *)
_class
loc:@fully_connected/biases
�
fully_connected/biases/AssignAssignfully_connected/biases(fully_connected/biases/Initializer/zeros*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
use_locking(
s
fully_connected/biases/readIdentityfully_connected/biases*
T0*)
_class
loc:@fully_connected/biases
|
fully_connected/MatMulMatMulstrided_slicefully_connected/weights/read*
T0*
transpose_a( *
transpose_b( 
w
fully_connected/BiasAddBiasAddfully_connected/MatMulfully_connected/biases/read*
data_formatNHWC*
T0
F
fully_connected/IdentityIdentityfully_connected/BiasAdd*
T0
<
subSubfully_connected/IdentityPlaceholder_1*
T0

SquareSquaresub*
T0
:
ConstConst*
valueB"       *
dtype0
?
SumSumSquareConst*

Tidx0*
	keep_dims( *
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
;
gradients/f_countConst*
value	B : *
dtype0
�
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_6/while/while_context
X
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N
J
gradients/SwitchSwitchgradients/Mergernn_6/while/LoopCond*
T0
P
gradients/Add/yConst^rnn_6/while/Identity*
value	B :*
dtype0
B
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0
�

gradients/NextIterationNextIterationgradients/Add>^gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPushV2>^gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPushV2:^gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPushV2:^gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPushV28^gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPushV2]^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
6
gradients/f_count_2Exitgradients/Switch*
T0
;
gradients/b_countConst*
dtype0*
value	B :
�
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
\
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N
`
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0
�
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
7
gradients/b_count_2LoopCondgradients/GreaterEqual
M
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0
Q
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0
�
gradients/NextIteration_1NextIterationgradients/SubX^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
8
gradients/b_count_3Exitgradients/Switch_1*
T0
=
gradients/f_count_3Const*
value	B : *
dtype0
�
gradients/f_count_4Entergradients/f_count_3*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant( 
\
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
T0*
N
N
gradients/Switch_2Switchgradients/Merge_2rnn_5/while/LoopCond*
T0
R
gradients/Add_1/yConst^rnn_5/while/Identity*
value	B :*
dtype0
H
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
T0
�

gradients/NextIteration_2NextIterationgradients/Add_1>^gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPushV2>^gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPushV2:^gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPushV2:^gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPushV28^gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPushV2]^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
8
gradients/f_count_5Exitgradients/Switch_2*
T0
=
gradients/b_count_4Const*
value	B :*
dtype0
�
gradients/b_count_5Entergradients/f_count_5*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
\
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
T0*
N
d
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0
�
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
9
gradients/b_count_6LoopCondgradients/GreaterEqual_1
M
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
T0
U
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
T0
�
gradients/NextIteration_3NextIterationgradients/Sub_1X^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
8
gradients/b_count_7Exitgradients/Switch_3*
T0
=
gradients/f_count_6Const*
value	B : *
dtype0
�
gradients/f_count_7Entergradients/f_count_6*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_4/while/while_context
\
gradients/Merge_4Mergegradients/f_count_7gradients/NextIteration_4*
T0*
N
N
gradients/Switch_4Switchgradients/Merge_4rnn_4/while/LoopCond*
T0
R
gradients/Add_2/yConst^rnn_4/while/Identity*
value	B :*
dtype0
H
gradients/Add_2Addgradients/Switch_4:1gradients/Add_2/y*
T0
�

gradients/NextIteration_4NextIterationgradients/Add_2>^gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPushV2>^gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPushV2:^gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPushV2:^gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPushV28^gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPushV2]^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
8
gradients/f_count_8Exitgradients/Switch_4*
T0
=
gradients/b_count_8Const*
dtype0*
value	B :
�
gradients/b_count_9Entergradients/f_count_8*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
\
gradients/Merge_5Mergegradients/b_count_9gradients/NextIteration_5*
T0*
N
d
gradients/GreaterEqual_2GreaterEqualgradients/Merge_5gradients/GreaterEqual_2/Enter*
T0
�
gradients/GreaterEqual_2/EnterEntergradients/b_count_8*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
:
gradients/b_count_10LoopCondgradients/GreaterEqual_2
N
gradients/Switch_5Switchgradients/Merge_5gradients/b_count_10*
T0
U
gradients/Sub_2Subgradients/Switch_5:1gradients/GreaterEqual_2/Enter*
T0
�
gradients/NextIteration_5NextIterationgradients/Sub_2X^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
9
gradients/b_count_11Exitgradients/Switch_5*
T0
=
gradients/f_count_9Const*
value	B : *
dtype0
�
gradients/f_count_10Entergradients/f_count_9*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_3/while/while_context
]
gradients/Merge_6Mergegradients/f_count_10gradients/NextIteration_6*
T0*
N
N
gradients/Switch_6Switchgradients/Merge_6rnn_3/while/LoopCond*
T0
R
gradients/Add_3/yConst^rnn_3/while/Identity*
value	B :*
dtype0
H
gradients/Add_3Addgradients/Switch_6:1gradients/Add_3/y*
T0
�

gradients/NextIteration_6NextIterationgradients/Add_3>^gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPushV2>^gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPushV2:^gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPushV2:^gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPushV28^gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPushV2]^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
9
gradients/f_count_11Exitgradients/Switch_6*
T0
>
gradients/b_count_12Const*
dtype0*
value	B :
�
gradients/b_count_13Entergradients/f_count_11*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant( 
]
gradients/Merge_7Mergegradients/b_count_13gradients/NextIteration_7*
T0*
N
d
gradients/GreaterEqual_3GreaterEqualgradients/Merge_7gradients/GreaterEqual_3/Enter*
T0
�
gradients/GreaterEqual_3/EnterEntergradients/b_count_12*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
:
gradients/b_count_14LoopCondgradients/GreaterEqual_3
N
gradients/Switch_7Switchgradients/Merge_7gradients/b_count_14*
T0
U
gradients/Sub_3Subgradients/Switch_7:1gradients/GreaterEqual_3/Enter*
T0
�
gradients/NextIteration_7NextIterationgradients/Sub_3X^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
9
gradients/b_count_15Exitgradients/Switch_7*
T0
>
gradients/f_count_12Const*
value	B : *
dtype0
�
gradients/f_count_13Entergradients/f_count_12*
T0*
is_constant( *
parallel_iterations *)

frame_namernn_2/while/while_context
]
gradients/Merge_8Mergegradients/f_count_13gradients/NextIteration_8*
T0*
N
N
gradients/Switch_8Switchgradients/Merge_8rnn_2/while/LoopCond*
T0
R
gradients/Add_4/yConst^rnn_2/while/Identity*
value	B :*
dtype0
H
gradients/Add_4Addgradients/Switch_8:1gradients/Add_4/y*
T0
�

gradients/NextIteration_8NextIterationgradients/Add_4>^gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPushV2>^gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPushV2:^gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPushV2:^gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPushV28^gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPushV2]^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
9
gradients/f_count_14Exitgradients/Switch_8*
T0
>
gradients/b_count_16Const*
value	B :*
dtype0
�
gradients/b_count_17Entergradients/f_count_14*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
]
gradients/Merge_9Mergegradients/b_count_17gradients/NextIteration_9*
T0*
N
d
gradients/GreaterEqual_4GreaterEqualgradients/Merge_9gradients/GreaterEqual_4/Enter*
T0
�
gradients/GreaterEqual_4/EnterEntergradients/b_count_16*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
:
gradients/b_count_18LoopCondgradients/GreaterEqual_4
N
gradients/Switch_9Switchgradients/Merge_9gradients/b_count_18*
T0
U
gradients/Sub_4Subgradients/Switch_9:1gradients/GreaterEqual_4/Enter*
T0
�
gradients/NextIteration_9NextIterationgradients/Sub_4X^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
9
gradients/b_count_19Exitgradients/Switch_9*
T0
>
gradients/f_count_15Const*
dtype0*
value	B : 
�
gradients/f_count_16Entergradients/f_count_15*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant( 
_
gradients/Merge_10Mergegradients/f_count_16gradients/NextIteration_10*
T0*
N
P
gradients/Switch_10Switchgradients/Merge_10rnn_1/while/LoopCond*
T0
R
gradients/Add_5/yConst^rnn_1/while/Identity*
value	B :*
dtype0
I
gradients/Add_5Addgradients/Switch_10:1gradients/Add_5/y*
T0
�

gradients/NextIteration_10NextIterationgradients/Add_5>^gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPushV2J^gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPushV2<^gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPushV2>^gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPushV2_1J^gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPushV2:^gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPushV2J^gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPushV2L^gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPushV2_18^gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPushV2:^gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPushV2H^gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPushV2J^gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPushV2_16^gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPushV28^gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPushV2]^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
:
gradients/f_count_17Exitgradients/Switch_10*
T0
>
gradients/b_count_20Const*
dtype0*
value	B :
�
gradients/b_count_21Entergradients/f_count_17*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
_
gradients/Merge_11Mergegradients/b_count_21gradients/NextIteration_11*
T0*
N
e
gradients/GreaterEqual_5GreaterEqualgradients/Merge_11gradients/GreaterEqual_5/Enter*
T0
�
gradients/GreaterEqual_5/EnterEntergradients/b_count_20*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
:
gradients/b_count_22LoopCondgradients/GreaterEqual_5
P
gradients/Switch_11Switchgradients/Merge_11gradients/b_count_22*
T0
V
gradients/Sub_5Subgradients/Switch_11:1gradients/GreaterEqual_5/Enter*
T0
�
gradients/NextIteration_11NextIterationgradients/Sub_5X^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
:
gradients/b_count_23Exitgradients/Switch_11*
T0
>
gradients/f_count_18Const*
value	B : *
dtype0
�
gradients/f_count_19Entergradients/f_count_18*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
_
gradients/Merge_12Mergegradients/f_count_19gradients/NextIteration_12*
T0*
N
N
gradients/Switch_12Switchgradients/Merge_12rnn/while/LoopCond*
T0
P
gradients/Add_6/yConst^rnn/while/Identity*
value	B :*
dtype0
I
gradients/Add_6Addgradients/Switch_12:1gradients/Add_6/y*
T0
�

gradients/NextIteration_12NextIterationgradients/Add_6:^gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPushV2F^gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPushV2H^gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPushV2_1D^gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPushV28^gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPushV2:^gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPushV2_1F^gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPushV2H^gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPushV2_14^gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPushV26^gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPushV2F^gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPushV2H^gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPushV2_14^gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPushV26^gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPushV2D^gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPushV2F^gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPushV2_12^gradients/rnn/while/LSTM/mul_grad/Mul/StackPushV24^gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPushV2[^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2*
T0
:
gradients/f_count_20Exitgradients/Switch_12*
T0
>
gradients/b_count_24Const*
value	B :*
dtype0
�
gradients/b_count_25Entergradients/f_count_20*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
_
gradients/Merge_13Mergegradients/b_count_25gradients/NextIteration_13*
N*
T0
e
gradients/GreaterEqual_6GreaterEqualgradients/Merge_13gradients/GreaterEqual_6/Enter*
T0
�
gradients/GreaterEqual_6/EnterEntergradients/b_count_24*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
:
gradients/b_count_26LoopCondgradients/GreaterEqual_6
P
gradients/Switch_13Switchgradients/Merge_13gradients/b_count_26*
T0
V
gradients/Sub_6Subgradients/Switch_13:1gradients/GreaterEqual_6/Enter*
T0
�
gradients/NextIteration_13NextIterationgradients/Sub_6V^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
:
gradients/b_count_27Exitgradients/Switch_13*
T0
U
 gradients/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0
n
gradients/Sum_grad/ReshapeReshapegradients/Fill gradients/Sum_grad/Reshape/shape*
T0*
Tshape0
B
gradients/Sum_grad/ShapeShapeSquare*
T0*
out_type0
p
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*

Tmultiples0*
T0
b
gradients/Square_grad/ConstConst^gradients/Sum_grad/Tile*
dtype0*
valueB
 *   @
K
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0
_
gradients/Square_grad/Mul_1Mulgradients/Sum_grad/Tilegradients/Square_grad/Mul*
T0
T
gradients/sub_grad/ShapeShapefully_connected/Identity*
T0*
out_type0
K
gradients/sub_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0
�
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
�
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
@
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0
r
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
�
2gradients/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0
�
7gradients/fully_connected/BiasAdd_grad/tuple/group_depsNoOp3^gradients/fully_connected/BiasAdd_grad/BiasAddGrad,^gradients/sub_grad/tuple/control_dependency
�
?gradients/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependency8^gradients/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
Agradients/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/fully_connected/BiasAdd_grad/BiasAddGrad8^gradients/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/fully_connected/BiasAdd_grad/BiasAddGrad
�
,gradients/fully_connected/MatMul_grad/MatMulMatMul?gradients/fully_connected/BiasAdd_grad/tuple/control_dependencyfully_connected/weights/read*
T0*
transpose_a( *
transpose_b(
�
.gradients/fully_connected/MatMul_grad/MatMul_1MatMulstrided_slice?gradients/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
6gradients/fully_connected/MatMul_grad/tuple/group_depsNoOp-^gradients/fully_connected/MatMul_grad/MatMul/^gradients/fully_connected/MatMul_grad/MatMul_1
�
>gradients/fully_connected/MatMul_grad/tuple/control_dependencyIdentity,gradients/fully_connected/MatMul_grad/MatMul7^gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/fully_connected/MatMul_grad/MatMul
�
@gradients/fully_connected/MatMul_grad/tuple/control_dependency_1Identity.gradients/fully_connected/MatMul_grad/MatMul_17^gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/fully_connected/MatMul_grad/MatMul_1
W
"gradients/strided_slice_grad/ShapeShapernn_6/transpose_1*
T0*
out_type0
�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad"gradients/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2>gradients/fully_connected/MatMul_grad/tuple/control_dependency*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
`
2gradients/rnn_6/transpose_1_grad/InvertPermutationInvertPermutationrnn_6/concat_2*
T0
�
*gradients/rnn_6/transpose_1_grad/transpose	Transpose-gradients/strided_slice_grad/StridedSliceGrad2gradients/rnn_6/transpose_1_grad/InvertPermutation*
T0*
Tperm0
�
[gradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_6/TensorArrayrnn_6/while/Exit_2*$
_class
loc:@rnn_6/TensorArray*
source	gradients
�
Wgradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_6/while/Exit_2\^gradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_6/TensorArray
�
agradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_6/TensorArrayStack/range*gradients/rnn_6/transpose_1_grad/transposeWgradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
>
gradients/zeros_like	ZerosLikernn_6/while/Exit_3*
T0
@
gradients/zeros_like_1	ZerosLikernn_6/while/Exit_4*
T0
�
(gradients/rnn_6/while/Exit_2_grad/b_exitEnteragradients/rnn_6/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
(gradients/rnn_6/while/Exit_3_grad/b_exitEntergradients/zeros_like*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
(gradients/rnn_6/while/Exit_4_grad/b_exitEntergradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
,gradients/rnn_6/while/Switch_2_grad/b_switchMerge(gradients/rnn_6/while/Exit_2_grad/b_exit3gradients/rnn_6/while/Switch_2_grad_1/NextIteration*
N*
T0
�
,gradients/rnn_6/while/Switch_3_grad/b_switchMerge(gradients/rnn_6/while/Exit_3_grad/b_exit3gradients/rnn_6/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_6/while/Switch_4_grad/b_switchMerge(gradients/rnn_6/while/Exit_4_grad/b_exit3gradients/rnn_6/while/Switch_4_grad_1/NextIteration*
T0*
N
�
)gradients/rnn_6/while/Merge_2_grad/SwitchSwitch,gradients/rnn_6/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_2_grad/b_switch
g
3gradients/rnn_6/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_6/while/Merge_2_grad/Switch
�
;gradients/rnn_6/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_6/while/Merge_2_grad/Switch4^gradients/rnn_6/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_2_grad/b_switch
�
=gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_6/while/Merge_2_grad/Switch:14^gradients/rnn_6/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_2_grad/b_switch
�
)gradients/rnn_6/while/Merge_3_grad/SwitchSwitch,gradients/rnn_6/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_3_grad/b_switch
g
3gradients/rnn_6/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_6/while/Merge_3_grad/Switch
�
;gradients/rnn_6/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_6/while/Merge_3_grad/Switch4^gradients/rnn_6/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_3_grad/b_switch
�
=gradients/rnn_6/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_6/while/Merge_3_grad/Switch:14^gradients/rnn_6/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_3_grad/b_switch
�
)gradients/rnn_6/while/Merge_4_grad/SwitchSwitch,gradients/rnn_6/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_4_grad/b_switch
g
3gradients/rnn_6/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_6/while/Merge_4_grad/Switch
�
;gradients/rnn_6/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_6/while/Merge_4_grad/Switch4^gradients/rnn_6/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_4_grad/b_switch
�
=gradients/rnn_6/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_6/while/Merge_4_grad/Switch:14^gradients/rnn_6/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_4_grad/b_switch
u
'gradients/rnn_6/while/Enter_2_grad/ExitExit;gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_6/while/Enter_3_grad/ExitExit;gradients/rnn_6/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_6/while/Enter_4_grad/ExitExit;gradients/rnn_6/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_6/while/LSTM_6/mul_2*
source	gradients
�
fgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_6/TensorArray*
T0*+
_class!
loc:@rnn_6/while/LSTM_6/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
\gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_6/while/LSTM_6/mul_2
�
Pgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*)
_class
loc:@rnn_6/while/Identity_1*
valueB :
���������*
dtype0
�
Vgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_6/while/Identity_1
�
Vgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
\gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_6/while/Identity_1^gradients/Add*
swap_memory( *
T0
�
[gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
agradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�

Wgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2=^gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV29^gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV29^gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPopV27^gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2\^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_6/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_2_grad/b_switch
�
gradients/AddNAddN=gradients/rnn_6/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
N*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_4_grad/b_switch
m
-gradients/rnn_6/while/LSTM_6/mul_2_grad/ShapeShapernn_6/while/LSTM_6/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape_1Shapernn_6/while/LSTM_6/Tanh_1*
T0*
out_type0
�
=gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape
�
Cgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Igradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Hgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Ngradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
Egradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape_1*
valueB :
���������
�
Egradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape_1*

stack_name 
�
Egradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Kgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_6/while/LSTM_6/mul_2_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Jgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Pgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
+gradients/rnn_6/while/LSTM_6/mul_2_grad/MulMulgradients/AddN6gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/ConstConst*
dtype0*,
_class"
 loc:@rnn_6/while/LSTM_6/Tanh_1*
valueB :
���������
�
1gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/f_accStackV21gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/Const*,
_class"
 loc:@rnn_6/while/LSTM_6/Tanh_1*

stack_name *
	elem_type0
�
1gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
7gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/Enterrnn_6/while/LSTM_6/Tanh_1^gradients/Add*
T0*
swap_memory( 
�
6gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
<gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
+gradients/rnn_6/while/LSTM_6/mul_2_grad/SumSum+gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul=gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_6/while/LSTM_6/mul_2_grad/ReshapeReshape+gradients/rnn_6/while/LSTM_6/mul_2_grad/SumHgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1Mul8gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2gradients/AddN*
T0
�
3gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_6/while/LSTM_6/Sigmoid_2*
valueB :
���������*
dtype0
�
3gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/Const*

stack_name *
	elem_type0*/
_class%
#!loc:@rnn_6/while/LSTM_6/Sigmoid_2
�
3gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
9gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/Enterrnn_6/while/LSTM_6/Sigmoid_2^gradients/Add*
T0*
swap_memory( 
�
8gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
>gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
-gradients/rnn_6/while/LSTM_6/mul_2_grad/Sum_1Sum-gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1?gradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape_1Reshape-gradients/rnn_6/while/LSTM_6/mul_2_grad/Sum_1Jgradients/rnn_6/while/LSTM_6/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape2^gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape_1
�
@gradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape9^gradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape
�
Bgradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape_19^gradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_6/while/LSTM_6/mul_2_grad/Reshape_1
�
7gradients/rnn_6/while/LSTM_6/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_6/while/LSTM_6/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_6/while/LSTM_6/mul_2_grad/Mul/StackPopV2Bgradients/rnn_6/while/LSTM_6/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_6/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_1AddN=gradients/rnn_6/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_6/while/LSTM_6/Tanh_1_grad/TanhGrad*
T0*?
_class5
31loc:@gradients/rnn_6/while/Switch_3_grad/b_switch*
N
g
-gradients/rnn_6/while/LSTM_6/add_1_grad/ShapeShapernn_6/while/LSTM_6/mul*
T0*
out_type0
k
/gradients/rnn_6/while/LSTM_6/add_1_grad/Shape_1Shapernn_6/while/LSTM_6/mul_1*
T0*
out_type0
�
=gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Shape*
valueB :
���������
�
Cgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
�
Igradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_6/while/LSTM_6/add_1_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Hgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Ngradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
Egradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Kgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_6/while/LSTM_6/add_1_grad/Shape_1^gradients/Add*
swap_memory( *
T0
�
Jgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Pgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
+gradients/rnn_6/while/LSTM_6/add_1_grad/SumSumgradients/AddN_1=gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_6/while/LSTM_6/add_1_grad/ReshapeReshape+gradients/rnn_6/while/LSTM_6/add_1_grad/SumHgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_6/while/LSTM_6/add_1_grad/Sum_1Sumgradients/AddN_1?gradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape_1Reshape-gradients/rnn_6/while/LSTM_6/add_1_grad/Sum_1Jgradients/rnn_6/while/LSTM_6/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape2^gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape_1
�
@gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape9^gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape
�
Bgradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape_19^gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_6/while/LSTM_6/add_1_grad/Reshape_1
i
+gradients/rnn_6/while/LSTM_6/mul_grad/ShapeShapernn_6/while/LSTM_6/Sigmoid*
T0*
out_type0
g
-gradients/rnn_6/while/LSTM_6/mul_grad/Shape_1Shapernn_6/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Ggradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_6/while/LSTM_6/mul_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Fgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Lgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
Cgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Const_1Const*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
Cgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Shape_1*

stack_name 
�
Cgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Igradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_6/while/LSTM_6/mul_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Hgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Ngradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
)gradients/rnn_6/while/LSTM_6/mul_grad/MulMul@gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependency4gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/ConstConst*
dtype0*)
_class
loc:@rnn_6/while/Identity_3*
valueB :
���������
�
/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/f_accStackV2/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/Const*)
_class
loc:@rnn_6/while/Identity_3*

stack_name *
	elem_type0
�
/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/EnterEnter/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/f_acc*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
�
5gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/Enterrnn_6/while/Identity_3^gradients/Add*
swap_memory( *
T0
�
4gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
:gradients/rnn_6/while/LSTM_6/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_6/while/LSTM_6/mul_grad/Mul/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
)gradients/rnn_6/while/LSTM_6/mul_grad/SumSum)gradients/rnn_6/while/LSTM_6/mul_grad/Mul;gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_6/while/LSTM_6/mul_grad/ReshapeReshape)gradients/rnn_6/while/LSTM_6/mul_grad/SumFgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1Mul6gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2@gradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_6/while/LSTM_6/Sigmoid*
valueB :
���������*
dtype0
�
1gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/f_accStackV21gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/Const*
	elem_type0*-
_class#
!loc:@rnn_6/while/LSTM_6/Sigmoid*

stack_name 
�
1gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/f_acc*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
�
7gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/Enterrnn_6/while/LSTM_6/Sigmoid^gradients/Add*
T0*
swap_memory( 
�
6gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
<gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
+gradients/rnn_6/while/LSTM_6/mul_grad/Sum_1Sum+gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1=gradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_6/while/LSTM_6/mul_grad/Reshape_1Reshape+gradients/rnn_6/while/LSTM_6/mul_grad/Sum_1Hgradients/rnn_6/while/LSTM_6/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_6/while/LSTM_6/mul_grad/tuple/group_depsNoOp.^gradients/rnn_6/while/LSTM_6/mul_grad/Reshape0^gradients/rnn_6/while/LSTM_6/mul_grad/Reshape_1
�
>gradients/rnn_6/while/LSTM_6/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_6/while/LSTM_6/mul_grad/Reshape7^gradients/rnn_6/while/LSTM_6/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Reshape
�
@gradients/rnn_6/while/LSTM_6/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_6/while/LSTM_6/mul_grad/Reshape_17^gradients/rnn_6/while/LSTM_6/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_grad/Reshape_1
m
-gradients/rnn_6/while/LSTM_6/mul_1_grad/ShapeShapernn_6/while/LSTM_6/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape_1Shapernn_6/while/LSTM_6/Tanh*
T0*
out_type0
�
=gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape*

stack_name 
�
Cgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Igradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Hgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Ngradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
Egradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape_1*
valueB :
���������
�
Egradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Kgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_6/while/LSTM_6/mul_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Jgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Pgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
+gradients/rnn_6/while/LSTM_6/mul_1_grad/MulMulBgradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependency_16gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/ConstConst**
_class 
loc:@rnn_6/while/LSTM_6/Tanh*
valueB :
���������*
dtype0
�
1gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/f_accStackV21gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/Const**
_class 
loc:@rnn_6/while/LSTM_6/Tanh*

stack_name *
	elem_type0
�
1gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
7gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/Enterrnn_6/while/LSTM_6/Tanh^gradients/Add*
T0*
swap_memory( 
�
6gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
<gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
+gradients/rnn_6/while/LSTM_6/mul_1_grad/SumSum+gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul=gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_6/while/LSTM_6/mul_1_grad/ReshapeReshape+gradients/rnn_6/while/LSTM_6/mul_1_grad/SumHgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1Mul8gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_6/while/LSTM_6/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/ConstConst*
dtype0*/
_class%
#!loc:@rnn_6/while/LSTM_6/Sigmoid_1*
valueB :
���������
�
3gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/Const*/
_class%
#!loc:@rnn_6/while/LSTM_6/Sigmoid_1*

stack_name *
	elem_type0
�
3gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
9gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/Enterrnn_6/while/LSTM_6/Sigmoid_1^gradients/Add*
T0*
swap_memory( 
�
8gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
>gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
-gradients/rnn_6/while/LSTM_6/mul_1_grad/Sum_1Sum-gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1?gradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape_1Reshape-gradients/rnn_6/while/LSTM_6/mul_1_grad/Sum_1Jgradients/rnn_6/while/LSTM_6/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape2^gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape_1
�
@gradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape9^gradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape
�
Bgradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape_19^gradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_6/while/LSTM_6/mul_1_grad/Reshape_1
�
5gradients/rnn_6/while/LSTM_6/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_6/while/LSTM_6/mul_grad/Mul_1/StackPopV2>gradients/rnn_6/while/LSTM_6/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_6/while/LSTM_6/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_6/while/LSTM_6/Tanh_grad/TanhGradTanhGrad6gradients/rnn_6/while/LSTM_6/mul_1_grad/Mul/StackPopV2Bgradients/rnn_6/while/LSTM_6/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_6/while/LSTM_6/add_grad/ShapeShapernn_6/while/LSTM_6/split:2*
T0*
out_type0
f
-gradients/rnn_6/while/LSTM_6/add_grad/Shape_1Const^gradients/Sub*
dtype0*
valueB 
�
;gradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_6/while/LSTM_6/add_grad/Shape_1*
T0
�
Agradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_6/while/LSTM_6/add_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_6/while/LSTM_6/add_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
Ggradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_6/while/LSTM_6/add_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Fgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Lgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
)gradients/rnn_6/while/LSTM_6/add_grad/SumSum5gradients/rnn_6/while/LSTM_6/Sigmoid_grad/SigmoidGrad;gradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn_6/while/LSTM_6/add_grad/ReshapeReshape)gradients/rnn_6/while/LSTM_6/add_grad/SumFgradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_6/while/LSTM_6/add_grad/Sum_1Sum5gradients/rnn_6/while/LSTM_6/Sigmoid_grad/SigmoidGrad=gradients/rnn_6/while/LSTM_6/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_6/while/LSTM_6/add_grad/Reshape_1Reshape+gradients/rnn_6/while/LSTM_6/add_grad/Sum_1-gradients/rnn_6/while/LSTM_6/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_6/while/LSTM_6/add_grad/tuple/group_depsNoOp.^gradients/rnn_6/while/LSTM_6/add_grad/Reshape0^gradients/rnn_6/while/LSTM_6/add_grad/Reshape_1
�
>gradients/rnn_6/while/LSTM_6/add_grad/tuple/control_dependencyIdentity-gradients/rnn_6/while/LSTM_6/add_grad/Reshape7^gradients/rnn_6/while/LSTM_6/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_6/while/LSTM_6/add_grad/Reshape
�
@gradients/rnn_6/while/LSTM_6/add_grad/tuple/control_dependency_1Identity/gradients/rnn_6/while/LSTM_6/add_grad/Reshape_17^gradients/rnn_6/while/LSTM_6/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/add_grad/Reshape_1
�
3gradients/rnn_6/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_6/while/LSTM_6/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_6/while/LSTM_6/split_grad/concatConcatV27gradients/rnn_6/while/LSTM_6/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_6/while/LSTM_6/Tanh_grad/TanhGrad>gradients/rnn_6/while/LSTM_6/add_grad/tuple/control_dependency7gradients/rnn_6/while/LSTM_6/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_6/while/LSTM_6/split_grad/concat/Const*
T0*
N*

Tidx0
n
4gradients/rnn_6/while/LSTM_6/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0
�
5gradients/rnn_6/while/LSTM_6/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_6/while/LSTM_6/split_grad/concat*
T0*
data_formatNHWC
�
:gradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_6/while/LSTM_6/BiasAdd_grad/BiasAddGrad/^gradients/rnn_6/while/LSTM_6/split_grad/concat
�
Bgradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_6/while/LSTM_6/split_grad/concat;^gradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_6/while/LSTM_6/split_grad/concat
�
Dgradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_6/while/LSTM_6/BiasAdd_grad/BiasAddGrad;^gradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_6/while/LSTM_6/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMulMatMulBgradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/control_dependency5gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *
transpose_b(
�
5gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul/EnterEnterrnn/LSTM_6/kernel/read*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
1gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1MatMul<gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
7gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/ConstConst*,
_class"
 loc:@rnn_6/while/LSTM_6/concat*
valueB :
���������*
dtype0
�
7gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/Const*
	elem_type0*,
_class"
 loc:@rnn_6/while/LSTM_6/concat*

stack_name 
�
7gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
=gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/Enterrnn_6/while/LSTM_6/concat^gradients/Add*
swap_memory( *
T0
�
<gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Bgradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
9gradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul2^gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1
�
Agradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul:^gradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul
�
Cgradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1:^gradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_6/while/LSTM_6/MatMul_grad/MatMul_1
g
5gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB�*    
�
7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
6gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0
�
3gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/AddAdd8gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_6/while/LSTM_6/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/Switch*
T0
h
.gradients/rnn_6/while/LSTM_6/concat_grad/ConstConst^gradients/Sub*
dtype0*
value	B :
g
-gradients/rnn_6/while/LSTM_6/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0
�
,gradients/rnn_6/while/LSTM_6/concat_grad/modFloorMod.gradients/rnn_6/while/LSTM_6/concat_grad/Const-gradients/rnn_6/while/LSTM_6/concat_grad/Rank*
T0
o
.gradients/rnn_6/while/LSTM_6/concat_grad/ShapeShapernn_6/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_6/while/LSTM_6/concat_grad/ShapeNShapeN:gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2<gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2_1*
N*
T0*
out_type0
�
5gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/ConstConst*0
_class&
$"loc:@rnn_6/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
5gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_accStackV25gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Const*
	elem_type0*0
_class&
$"loc:@rnn_6/while/TensorArrayReadV3*

stack_name 
�
5gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/EnterEnter5gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_acc*
parallel_iterations *)

frame_namernn_6/while/while_context*
T0*
is_constant(
�
;gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Enterrnn_6/while/TensorArrayReadV3^gradients/Add*
T0*
swap_memory( 
�
:gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
@gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
7gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Const_1Const*)
_class
loc:@rnn_6/while/Identity_4*
valueB :
���������*
dtype0
�
7gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Const_1*)
_class
loc:@rnn_6/while/Identity_4*

stack_name *
	elem_type0
�
7gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_6/while/while_context
�
=gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/Enter_1rnn_6/while/Identity_4^gradients/Add*
T0*
swap_memory( 
�
<gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Bgradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant(
�
5gradients/rnn_6/while/LSTM_6/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_6/while/LSTM_6/concat_grad/mod/gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN1gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN:1*
N
�
.gradients/rnn_6/while/LSTM_6/concat_grad/SliceSliceAgradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/control_dependency5gradients/rnn_6/while/LSTM_6/concat_grad/ConcatOffset/gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_6/while/LSTM_6/concat_grad/Slice_1SliceAgradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/control_dependency7gradients/rnn_6/while/LSTM_6/concat_grad/ConcatOffset:11gradients/rnn_6/while/LSTM_6/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_6/while/LSTM_6/concat_grad/tuple/group_depsNoOp/^gradients/rnn_6/while/LSTM_6/concat_grad/Slice1^gradients/rnn_6/while/LSTM_6/concat_grad/Slice_1
�
Agradients/rnn_6/while/LSTM_6/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_6/while/LSTM_6/concat_grad/Slice:^gradients/rnn_6/while/LSTM_6/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_6/while/LSTM_6/concat_grad/Slice
�
Cgradients/rnn_6/while/LSTM_6/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_6/while/LSTM_6/concat_grad/Slice_1:^gradients/rnn_6/while/LSTM_6/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_6/while/LSTM_6/concat_grad/Slice_1
k
4gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context*
T0*
is_constant( 
�
6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_1<gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/NextIteration*
T0*
N
�
5gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0
�
2gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/AddAdd7gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/Switch:1Cgradients/rnn_6/while/LSTM_6/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*6
_class,
*(loc:@rnn_6/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_6/TensorArray_1*
T0*6
_class,
*(loc:@rnn_6/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
Vgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6
_class,
*(loc:@rnn_6/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
Jgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_6/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_6/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_6/while/LSTM_6/concat_grad/tuple/control_dependencyJgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_6/while/while_context
�
<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
;gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0
�
8gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_6/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_6/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_6/while/LSTM_6/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_6/TensorArray_1<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_6/TensorArray_1*
source	gradients
�
mgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_6/TensorArray_1
�
cgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_6/TensorArrayUnstack/rangemgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0
�
`gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_6/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_6/transpose_grad/InvertPermutationInvertPermutationrnn_6/concat*
T0
�
(gradients/rnn_6/transpose_grad/transpose	Transposehgradients/rnn_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_6/transpose_grad/InvertPermutation*
Tperm0*
T0
`
2gradients/rnn_5/transpose_1_grad/InvertPermutationInvertPermutationrnn_5/concat_2*
T0
�
*gradients/rnn_5/transpose_1_grad/transpose	Transpose(gradients/rnn_6/transpose_grad/transpose2gradients/rnn_5/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
[gradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_5/TensorArrayrnn_5/while/Exit_2*$
_class
loc:@rnn_5/TensorArray*
source	gradients
�
Wgradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_5/while/Exit_2\^gradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_5/TensorArray
�
agradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_5/TensorArrayStack/range*gradients/rnn_5/transpose_1_grad/transposeWgradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
@
gradients/zeros_like_2	ZerosLikernn_5/while/Exit_3*
T0
@
gradients/zeros_like_3	ZerosLikernn_5/while/Exit_4*
T0
�
(gradients/rnn_5/while/Exit_2_grad/b_exitEnteragradients/rnn_5/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
(gradients/rnn_5/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant( 
�
(gradients/rnn_5/while/Exit_4_grad/b_exitEntergradients/zeros_like_3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
,gradients/rnn_5/while/Switch_2_grad/b_switchMerge(gradients/rnn_5/while/Exit_2_grad/b_exit3gradients/rnn_5/while/Switch_2_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_5/while/Switch_3_grad/b_switchMerge(gradients/rnn_5/while/Exit_3_grad/b_exit3gradients/rnn_5/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_5/while/Switch_4_grad/b_switchMerge(gradients/rnn_5/while/Exit_4_grad/b_exit3gradients/rnn_5/while/Switch_4_grad_1/NextIteration*
T0*
N
�
)gradients/rnn_5/while/Merge_2_grad/SwitchSwitch,gradients/rnn_5/while/Switch_2_grad/b_switchgradients/b_count_6*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_2_grad/b_switch
g
3gradients/rnn_5/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_5/while/Merge_2_grad/Switch
�
;gradients/rnn_5/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_5/while/Merge_2_grad/Switch4^gradients/rnn_5/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_2_grad/b_switch
�
=gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_5/while/Merge_2_grad/Switch:14^gradients/rnn_5/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_2_grad/b_switch
�
)gradients/rnn_5/while/Merge_3_grad/SwitchSwitch,gradients/rnn_5/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_3_grad/b_switch
g
3gradients/rnn_5/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_5/while/Merge_3_grad/Switch
�
;gradients/rnn_5/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_5/while/Merge_3_grad/Switch4^gradients/rnn_5/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_3_grad/b_switch
�
=gradients/rnn_5/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_5/while/Merge_3_grad/Switch:14^gradients/rnn_5/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_3_grad/b_switch
�
)gradients/rnn_5/while/Merge_4_grad/SwitchSwitch,gradients/rnn_5/while/Switch_4_grad/b_switchgradients/b_count_6*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_4_grad/b_switch
g
3gradients/rnn_5/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_5/while/Merge_4_grad/Switch
�
;gradients/rnn_5/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_5/while/Merge_4_grad/Switch4^gradients/rnn_5/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_4_grad/b_switch
�
=gradients/rnn_5/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_5/while/Merge_4_grad/Switch:14^gradients/rnn_5/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_4_grad/b_switch
u
'gradients/rnn_5/while/Enter_2_grad/ExitExit;gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_5/while/Enter_3_grad/ExitExit;gradients/rnn_5/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_5/while/Enter_4_grad/ExitExit;gradients/rnn_5/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_5/while/LSTM_5/mul_2*
source	gradients
�
fgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_5/TensorArray*
T0*+
_class!
loc:@rnn_5/while/LSTM_5/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
\gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_5/while/LSTM_5/mul_2
�
Pgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
dtype0*)
_class
loc:@rnn_5/while/Identity_1*
valueB :
���������
�
Vgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_5/while/Identity_1
�
Vgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
\gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_5/while/Identity_1^gradients/Add_1*
T0*
swap_memory( 
�
[gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
agradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�

Wgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2=^gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV29^gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV29^gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPopV27^gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2\^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_5/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_2_grad/b_switch
�
gradients/AddN_2AddN=gradients/rnn_5/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_4_grad/b_switch*
N
m
-gradients/rnn_5/while/LSTM_5/mul_2_grad/ShapeShapernn_5/while/LSTM_5/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape_1Shapernn_5/while/LSTM_5/Tanh_1*
T0*
out_type0
�
=gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape*
valueB :
���������
�
Cgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape*

stack_name 
�
Cgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Igradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape^gradients/Add_1*
T0*
swap_memory( 
�
Hgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Ngradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Egradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape_1*

stack_name 
�
Egradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Kgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_5/while/LSTM_5/mul_2_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( 
�
Jgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0
�
Pgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
+gradients/rnn_5/while/LSTM_5/mul_2_grad/MulMulgradients/AddN_26gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/ConstConst*
dtype0*,
_class"
 loc:@rnn_5/while/LSTM_5/Tanh_1*
valueB :
���������
�
1gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/f_accStackV21gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/Const*,
_class"
 loc:@rnn_5/while/LSTM_5/Tanh_1*

stack_name *
	elem_type0
�
1gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
7gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/Enterrnn_5/while/LSTM_5/Tanh_1^gradients/Add_1*
T0*
swap_memory( 
�
6gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
<gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
+gradients/rnn_5/while/LSTM_5/mul_2_grad/SumSum+gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul=gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_5/while/LSTM_5/mul_2_grad/ReshapeReshape+gradients/rnn_5/while/LSTM_5/mul_2_grad/SumHgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1Mul8gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2gradients/AddN_2*
T0
�
3gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_5/while/LSTM_5/Sigmoid_2*
valueB :
���������*
dtype0
�
3gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/Const*/
_class%
#!loc:@rnn_5/while/LSTM_5/Sigmoid_2*

stack_name *
	elem_type0
�
3gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
9gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/Enterrnn_5/while/LSTM_5/Sigmoid_2^gradients/Add_1*
T0*
swap_memory( 
�
8gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
>gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant(
�
-gradients/rnn_5/while/LSTM_5/mul_2_grad/Sum_1Sum-gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1?gradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
1gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape_1Reshape-gradients/rnn_5/while/LSTM_5/mul_2_grad/Sum_1Jgradients/rnn_5/while/LSTM_5/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape2^gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape_1
�
@gradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape9^gradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape
�
Bgradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape_19^gradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_5/while/LSTM_5/mul_2_grad/Reshape_1
�
7gradients/rnn_5/while/LSTM_5/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_5/while/LSTM_5/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_5/while/LSTM_5/mul_2_grad/Mul/StackPopV2Bgradients/rnn_5/while/LSTM_5/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_5/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_3AddN=gradients/rnn_5/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_5/while/LSTM_5/Tanh_1_grad/TanhGrad*
T0*?
_class5
31loc:@gradients/rnn_5/while/Switch_3_grad/b_switch*
N
g
-gradients/rnn_5/while/LSTM_5/add_1_grad/ShapeShapernn_5/while/LSTM_5/mul*
T0*
out_type0
k
/gradients/rnn_5/while/LSTM_5/add_1_grad/Shape_1Shapernn_5/while/LSTM_5/mul_1*
T0*
out_type0
�
=gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Igradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_5/while/LSTM_5/add_1_grad/Shape^gradients/Add_1*
T0*
swap_memory( 
�
Hgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Ngradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Egradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Kgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_5/while/LSTM_5/add_1_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( 
�
Jgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0
�
Pgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
+gradients/rnn_5/while/LSTM_5/add_1_grad/SumSumgradients/AddN_3=gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_5/while/LSTM_5/add_1_grad/ReshapeReshape+gradients/rnn_5/while/LSTM_5/add_1_grad/SumHgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_5/while/LSTM_5/add_1_grad/Sum_1Sumgradients/AddN_3?gradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape_1Reshape-gradients/rnn_5/while/LSTM_5/add_1_grad/Sum_1Jgradients/rnn_5/while/LSTM_5/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape2^gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape_1
�
@gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape9^gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape
�
Bgradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape_19^gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_5/while/LSTM_5/add_1_grad/Reshape_1
i
+gradients/rnn_5/while/LSTM_5/mul_grad/ShapeShapernn_5/while/LSTM_5/Sigmoid*
T0*
out_type0
g
-gradients/rnn_5/while/LSTM_5/mul_grad/Shape_1Shapernn_5/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*>
_class4
20loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Shape
�
Agradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Ggradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_5/while/LSTM_5/mul_grad/Shape^gradients/Add_1*
T0*
swap_memory( 
�
Fgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Lgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Cgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Const_1Const*
dtype0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Shape_1*
valueB :
���������
�
Cgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Shape_1*

stack_name 
�
Cgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant(
�
Igradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_5/while/LSTM_5/mul_grad/Shape_1^gradients/Add_1*
swap_memory( *
T0
�
Hgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0
�
Ngradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant(
�
)gradients/rnn_5/while/LSTM_5/mul_grad/MulMul@gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependency4gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/ConstConst*)
_class
loc:@rnn_5/while/Identity_3*
valueB :
���������*
dtype0
�
/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/f_accStackV2/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_5/while/Identity_3
�
/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/EnterEnter/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
5gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/Enterrnn_5/while/Identity_3^gradients/Add_1*
swap_memory( *
T0
�
4gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
:gradients/rnn_5/while/LSTM_5/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_5/while/LSTM_5/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
)gradients/rnn_5/while/LSTM_5/mul_grad/SumSum)gradients/rnn_5/while/LSTM_5/mul_grad/Mul;gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn_5/while/LSTM_5/mul_grad/ReshapeReshape)gradients/rnn_5/while/LSTM_5/mul_grad/SumFgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1Mul6gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2@gradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/ConstConst*
dtype0*-
_class#
!loc:@rnn_5/while/LSTM_5/Sigmoid*
valueB :
���������
�
1gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/f_accStackV21gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/Const*
	elem_type0*-
_class#
!loc:@rnn_5/while/LSTM_5/Sigmoid*

stack_name 
�
1gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
7gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/Enterrnn_5/while/LSTM_5/Sigmoid^gradients/Add_1*
T0*
swap_memory( 
�
6gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
<gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant(
�
+gradients/rnn_5/while/LSTM_5/mul_grad/Sum_1Sum+gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1=gradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_5/while/LSTM_5/mul_grad/Reshape_1Reshape+gradients/rnn_5/while/LSTM_5/mul_grad/Sum_1Hgradients/rnn_5/while/LSTM_5/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_5/while/LSTM_5/mul_grad/tuple/group_depsNoOp.^gradients/rnn_5/while/LSTM_5/mul_grad/Reshape0^gradients/rnn_5/while/LSTM_5/mul_grad/Reshape_1
�
>gradients/rnn_5/while/LSTM_5/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_5/while/LSTM_5/mul_grad/Reshape7^gradients/rnn_5/while/LSTM_5/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Reshape
�
@gradients/rnn_5/while/LSTM_5/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_5/while/LSTM_5/mul_grad/Reshape_17^gradients/rnn_5/while/LSTM_5/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_grad/Reshape_1
m
-gradients/rnn_5/while/LSTM_5/mul_1_grad/ShapeShapernn_5/while/LSTM_5/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape_1Shapernn_5/while/LSTM_5/Tanh*
T0*
out_type0
�
=gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape*

stack_name 
�
Cgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Igradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape^gradients/Add_1*
swap_memory( *
T0
�
Hgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Ngradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Egradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape_1*

stack_name 
�
Egradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Kgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_5/while/LSTM_5/mul_1_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( 
�
Jgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0
�
Pgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
+gradients/rnn_5/while/LSTM_5/mul_1_grad/MulMulBgradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependency_16gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/ConstConst*
dtype0**
_class 
loc:@rnn_5/while/LSTM_5/Tanh*
valueB :
���������
�
1gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/f_accStackV21gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/Const**
_class 
loc:@rnn_5/while/LSTM_5/Tanh*

stack_name *
	elem_type0
�
1gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/f_acc*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant(
�
7gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/Enterrnn_5/while/LSTM_5/Tanh^gradients/Add_1*
T0*
swap_memory( 
�
6gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
<gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
+gradients/rnn_5/while/LSTM_5/mul_1_grad/SumSum+gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul=gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_5/while/LSTM_5/mul_1_grad/ReshapeReshape+gradients/rnn_5/while/LSTM_5/mul_1_grad/SumHgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1Mul8gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_5/while/LSTM_5/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_5/while/LSTM_5/Sigmoid_1*
valueB :
���������*
dtype0
�
3gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/Const*/
_class%
#!loc:@rnn_5/while/LSTM_5/Sigmoid_1*

stack_name *
	elem_type0
�
3gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/f_acc*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant(
�
9gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/Enterrnn_5/while/LSTM_5/Sigmoid_1^gradients/Add_1*
swap_memory( *
T0
�
8gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
>gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
-gradients/rnn_5/while/LSTM_5/mul_1_grad/Sum_1Sum-gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1?gradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
1gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape_1Reshape-gradients/rnn_5/while/LSTM_5/mul_1_grad/Sum_1Jgradients/rnn_5/while/LSTM_5/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape2^gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape_1
�
@gradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape9^gradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape
�
Bgradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape_19^gradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_5/while/LSTM_5/mul_1_grad/Reshape_1
�
5gradients/rnn_5/while/LSTM_5/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_5/while/LSTM_5/mul_grad/Mul_1/StackPopV2>gradients/rnn_5/while/LSTM_5/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_5/while/LSTM_5/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_5/while/LSTM_5/Tanh_grad/TanhGradTanhGrad6gradients/rnn_5/while/LSTM_5/mul_1_grad/Mul/StackPopV2Bgradients/rnn_5/while/LSTM_5/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_5/while/LSTM_5/add_grad/ShapeShapernn_5/while/LSTM_5/split:2*
T0*
out_type0
h
-gradients/rnn_5/while/LSTM_5/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0
�
;gradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_5/while/LSTM_5/add_grad/Shape_1*
T0
�
Agradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/ConstConst*
dtype0*>
_class4
20loc:@gradients/rnn_5/while/LSTM_5/add_grad/Shape*
valueB :
���������
�
Agradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_5/while/LSTM_5/add_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
Ggradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_5/while/LSTM_5/add_grad/Shape^gradients/Add_1*
swap_memory( *
T0
�
Fgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Lgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
)gradients/rnn_5/while/LSTM_5/add_grad/SumSum5gradients/rnn_5/while/LSTM_5/Sigmoid_grad/SigmoidGrad;gradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_5/while/LSTM_5/add_grad/ReshapeReshape)gradients/rnn_5/while/LSTM_5/add_grad/SumFgradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_5/while/LSTM_5/add_grad/Sum_1Sum5gradients/rnn_5/while/LSTM_5/Sigmoid_grad/SigmoidGrad=gradients/rnn_5/while/LSTM_5/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_5/while/LSTM_5/add_grad/Reshape_1Reshape+gradients/rnn_5/while/LSTM_5/add_grad/Sum_1-gradients/rnn_5/while/LSTM_5/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_5/while/LSTM_5/add_grad/tuple/group_depsNoOp.^gradients/rnn_5/while/LSTM_5/add_grad/Reshape0^gradients/rnn_5/while/LSTM_5/add_grad/Reshape_1
�
>gradients/rnn_5/while/LSTM_5/add_grad/tuple/control_dependencyIdentity-gradients/rnn_5/while/LSTM_5/add_grad/Reshape7^gradients/rnn_5/while/LSTM_5/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_5/while/LSTM_5/add_grad/Reshape
�
@gradients/rnn_5/while/LSTM_5/add_grad/tuple/control_dependency_1Identity/gradients/rnn_5/while/LSTM_5/add_grad/Reshape_17^gradients/rnn_5/while/LSTM_5/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/add_grad/Reshape_1
�
3gradients/rnn_5/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_5/while/LSTM_5/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_5/while/LSTM_5/split_grad/concatConcatV27gradients/rnn_5/while/LSTM_5/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_5/while/LSTM_5/Tanh_grad/TanhGrad>gradients/rnn_5/while/LSTM_5/add_grad/tuple/control_dependency7gradients/rnn_5/while/LSTM_5/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_5/while/LSTM_5/split_grad/concat/Const*
T0*
N*

Tidx0
p
4gradients/rnn_5/while/LSTM_5/split_grad/concat/ConstConst^gradients/Sub_1*
value	B :*
dtype0
�
5gradients/rnn_5/while/LSTM_5/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_5/while/LSTM_5/split_grad/concat*
T0*
data_formatNHWC
�
:gradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_5/while/LSTM_5/BiasAdd_grad/BiasAddGrad/^gradients/rnn_5/while/LSTM_5/split_grad/concat
�
Bgradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_5/while/LSTM_5/split_grad/concat;^gradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_5/while/LSTM_5/split_grad/concat
�
Dgradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_5/while/LSTM_5/BiasAdd_grad/BiasAddGrad;^gradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_5/while/LSTM_5/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMulMatMulBgradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/control_dependency5gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( 
�
5gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul/EnterEnterrnn/LSTM_5/kernel/read*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant(
�
1gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1MatMul<gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
7gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/ConstConst*
dtype0*,
_class"
 loc:@rnn_5/while/LSTM_5/concat*
valueB :
���������
�
7gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/Const*,
_class"
 loc:@rnn_5/while/LSTM_5/concat*

stack_name *
	elem_type0
�
7gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
=gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/Enterrnn_5/while/LSTM_5/concat^gradients/Add_1*
T0*
swap_memory( 
�
<gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
Bgradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
9gradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul2^gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1
�
Agradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul:^gradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul
�
Cgradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1:^gradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_5/while/LSTM_5/MatMul_grad/MatMul_1
g
5gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
6gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*
T0
�
3gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/AddAdd8gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_5/while/LSTM_5/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/Switch*
T0
j
.gradients/rnn_5/while/LSTM_5/concat_grad/ConstConst^gradients/Sub_1*
value	B :*
dtype0
i
-gradients/rnn_5/while/LSTM_5/concat_grad/RankConst^gradients/Sub_1*
dtype0*
value	B :
�
,gradients/rnn_5/while/LSTM_5/concat_grad/modFloorMod.gradients/rnn_5/while/LSTM_5/concat_grad/Const-gradients/rnn_5/while/LSTM_5/concat_grad/Rank*
T0
o
.gradients/rnn_5/while/LSTM_5/concat_grad/ShapeShapernn_5/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_5/while/LSTM_5/concat_grad/ShapeNShapeN:gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2<gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2_1*
N*
T0*
out_type0
�
5gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/ConstConst*0
_class&
$"loc:@rnn_5/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
5gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_accStackV25gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Const*0
_class&
$"loc:@rnn_5/while/TensorArrayReadV3*

stack_name *
	elem_type0
�
5gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/EnterEnter5gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_5/while/while_context
�
;gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Enterrnn_5/while/TensorArrayReadV3^gradients/Add_1*
T0*
swap_memory( 
�
:gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
�
@gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
7gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Const_1Const*
dtype0*)
_class
loc:@rnn_5/while/Identity_4*
valueB :
���������
�
7gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Const_1*
	elem_type0*)
_class
loc:@rnn_5/while/Identity_4*

stack_name 
�
7gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_acc_1*
parallel_iterations *)

frame_namernn_5/while/while_context*
T0*
is_constant(
�
=gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/Enter_1rnn_5/while/Identity_4^gradients/Add_1*
T0*
swap_memory( 
�
<gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0
�
Bgradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant(
�
5gradients/rnn_5/while/LSTM_5/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_5/while/LSTM_5/concat_grad/mod/gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN1gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN:1*
N
�
.gradients/rnn_5/while/LSTM_5/concat_grad/SliceSliceAgradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/control_dependency5gradients/rnn_5/while/LSTM_5/concat_grad/ConcatOffset/gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_5/while/LSTM_5/concat_grad/Slice_1SliceAgradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/control_dependency7gradients/rnn_5/while/LSTM_5/concat_grad/ConcatOffset:11gradients/rnn_5/while/LSTM_5/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_5/while/LSTM_5/concat_grad/tuple/group_depsNoOp/^gradients/rnn_5/while/LSTM_5/concat_grad/Slice1^gradients/rnn_5/while/LSTM_5/concat_grad/Slice_1
�
Agradients/rnn_5/while/LSTM_5/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_5/while/LSTM_5/concat_grad/Slice:^gradients/rnn_5/while/LSTM_5/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_5/while/LSTM_5/concat_grad/Slice
�
Cgradients/rnn_5/while/LSTM_5/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_5/while/LSTM_5/concat_grad/Slice_1:^gradients/rnn_5/while/LSTM_5/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_5/while/LSTM_5/concat_grad/Slice_1
k
4gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_1<gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/NextIteration*
T0*
N
�
5gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0
�
2gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/AddAdd7gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/Switch:1Cgradients/rnn_5/while/LSTM_5/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*6
_class,
*(loc:@rnn_5/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_5/TensorArray_1*
T0*6
_class,
*(loc:@rnn_5/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Vgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6
_class,
*(loc:@rnn_5/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context
�
Jgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_5/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_5/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_5/while/LSTM_5/concat_grad/tuple/control_dependencyJgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *3

frame_name%#gradients/rnn_5/while/while_context*
T0*
is_constant( 
�
<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
;gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
T0
�
8gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_5/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_5/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_5/while/LSTM_5/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_5/TensorArray_1<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_5/TensorArray_1*
source	gradients
�
mgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_5/TensorArray_1
�
cgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_5/TensorArrayUnstack/rangemgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
element_shape:
�
`gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_5/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_5/transpose_grad/InvertPermutationInvertPermutationrnn_5/concat*
T0
�
(gradients/rnn_5/transpose_grad/transpose	Transposehgradients/rnn_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_5/transpose_grad/InvertPermutation*
T0*
Tperm0
`
2gradients/rnn_4/transpose_1_grad/InvertPermutationInvertPermutationrnn_4/concat_2*
T0
�
*gradients/rnn_4/transpose_1_grad/transpose	Transpose(gradients/rnn_5/transpose_grad/transpose2gradients/rnn_4/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
[gradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_4/TensorArrayrnn_4/while/Exit_2*$
_class
loc:@rnn_4/TensorArray*
source	gradients
�
Wgradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_4/while/Exit_2\^gradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_4/TensorArray
�
agradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_4/TensorArrayStack/range*gradients/rnn_4/transpose_1_grad/transposeWgradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
@
gradients/zeros_like_4	ZerosLikernn_4/while/Exit_3*
T0
@
gradients/zeros_like_5	ZerosLikernn_4/while/Exit_4*
T0
�
(gradients/rnn_4/while/Exit_2_grad/b_exitEnteragradients/rnn_4/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
(gradients/rnn_4/while/Exit_3_grad/b_exitEntergradients/zeros_like_4*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant( 
�
(gradients/rnn_4/while/Exit_4_grad/b_exitEntergradients/zeros_like_5*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
,gradients/rnn_4/while/Switch_2_grad/b_switchMerge(gradients/rnn_4/while/Exit_2_grad/b_exit3gradients/rnn_4/while/Switch_2_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_4/while/Switch_3_grad/b_switchMerge(gradients/rnn_4/while/Exit_3_grad/b_exit3gradients/rnn_4/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_4/while/Switch_4_grad/b_switchMerge(gradients/rnn_4/while/Exit_4_grad/b_exit3gradients/rnn_4/while/Switch_4_grad_1/NextIteration*
N*
T0
�
)gradients/rnn_4/while/Merge_2_grad/SwitchSwitch,gradients/rnn_4/while/Switch_2_grad/b_switchgradients/b_count_10*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_2_grad/b_switch
g
3gradients/rnn_4/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_4/while/Merge_2_grad/Switch
�
;gradients/rnn_4/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_4/while/Merge_2_grad/Switch4^gradients/rnn_4/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_2_grad/b_switch
�
=gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_4/while/Merge_2_grad/Switch:14^gradients/rnn_4/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_2_grad/b_switch
�
)gradients/rnn_4/while/Merge_3_grad/SwitchSwitch,gradients/rnn_4/while/Switch_3_grad/b_switchgradients/b_count_10*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_3_grad/b_switch
g
3gradients/rnn_4/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_4/while/Merge_3_grad/Switch
�
;gradients/rnn_4/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_4/while/Merge_3_grad/Switch4^gradients/rnn_4/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_3_grad/b_switch
�
=gradients/rnn_4/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_4/while/Merge_3_grad/Switch:14^gradients/rnn_4/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_3_grad/b_switch
�
)gradients/rnn_4/while/Merge_4_grad/SwitchSwitch,gradients/rnn_4/while/Switch_4_grad/b_switchgradients/b_count_10*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_4_grad/b_switch
g
3gradients/rnn_4/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_4/while/Merge_4_grad/Switch
�
;gradients/rnn_4/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_4/while/Merge_4_grad/Switch4^gradients/rnn_4/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_4_grad/b_switch
�
=gradients/rnn_4/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_4/while/Merge_4_grad/Switch:14^gradients/rnn_4/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_4_grad/b_switch
u
'gradients/rnn_4/while/Enter_2_grad/ExitExit;gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_4/while/Enter_3_grad/ExitExit;gradients/rnn_4/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_4/while/Enter_4_grad/ExitExit;gradients/rnn_4/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_4/while/LSTM_4/mul_2*
source	gradients
�
fgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_4/TensorArray*
T0*+
_class!
loc:@rnn_4/while/LSTM_4/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
\gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_4/while/LSTM_4/mul_2
�
Pgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*)
_class
loc:@rnn_4/while/Identity_1*
valueB :
���������*
dtype0
�
Vgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*)
_class
loc:@rnn_4/while/Identity_1*

stack_name *
	elem_type0
�
Vgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
�
\gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_4/while/Identity_1^gradients/Add_2*
T0*
swap_memory( 
�
[gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
agradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�

Wgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2=^gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV29^gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV29^gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPopV27^gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2\^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_4/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_2_grad/b_switch
�
gradients/AddN_4AddN=gradients/rnn_4/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
N*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_4_grad/b_switch
m
-gradients/rnn_4/while/LSTM_4/mul_2_grad/ShapeShapernn_4/while/LSTM_4/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape_1Shapernn_4/while/LSTM_4/Tanh_1*
T0*
out_type0
�
=gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Igradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape^gradients/Add_2*
T0*
swap_memory( 
�
Hgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Ngradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
Egradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Const_1*

stack_name *
	elem_type0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape_1
�
Egradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Kgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_4/while/LSTM_4/mul_2_grad/Shape_1^gradients/Add_2*
swap_memory( *
T0
�
Jgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0
�
Pgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
+gradients/rnn_4/while/LSTM_4/mul_2_grad/MulMulgradients/AddN_46gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/ConstConst*,
_class"
 loc:@rnn_4/while/LSTM_4/Tanh_1*
valueB :
���������*
dtype0
�
1gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/f_accStackV21gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/Const*,
_class"
 loc:@rnn_4/while/LSTM_4/Tanh_1*

stack_name *
	elem_type0
�
1gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
7gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/Enterrnn_4/while/LSTM_4/Tanh_1^gradients/Add_2*
swap_memory( *
T0
�
6gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
<gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
+gradients/rnn_4/while/LSTM_4/mul_2_grad/SumSum+gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul=gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_4/while/LSTM_4/mul_2_grad/ReshapeReshape+gradients/rnn_4/while/LSTM_4/mul_2_grad/SumHgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1Mul8gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2gradients/AddN_4*
T0
�
3gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_4/while/LSTM_4/Sigmoid_2*
valueB :
���������*
dtype0
�
3gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/Const*

stack_name *
	elem_type0*/
_class%
#!loc:@rnn_4/while/LSTM_4/Sigmoid_2
�
3gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/f_acc*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
�
9gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/Enterrnn_4/while/LSTM_4/Sigmoid_2^gradients/Add_2*
T0*
swap_memory( 
�
8gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
>gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
-gradients/rnn_4/while/LSTM_4/mul_2_grad/Sum_1Sum-gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1?gradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape_1Reshape-gradients/rnn_4/while/LSTM_4/mul_2_grad/Sum_1Jgradients/rnn_4/while/LSTM_4/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape2^gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape_1
�
@gradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape9^gradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape
�
Bgradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape_19^gradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_4/while/LSTM_4/mul_2_grad/Reshape_1
�
7gradients/rnn_4/while/LSTM_4/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_4/while/LSTM_4/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_4/while/LSTM_4/mul_2_grad/Mul/StackPopV2Bgradients/rnn_4/while/LSTM_4/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_4/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_5AddN=gradients/rnn_4/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_4/while/LSTM_4/Tanh_1_grad/TanhGrad*
N*
T0*?
_class5
31loc:@gradients/rnn_4/while/Switch_3_grad/b_switch
g
-gradients/rnn_4/while/LSTM_4/add_1_grad/ShapeShapernn_4/while/LSTM_4/mul*
T0*
out_type0
k
/gradients/rnn_4/while/LSTM_4/add_1_grad/Shape_1Shapernn_4/while/LSTM_4/mul_1*
T0*
out_type0
�
=gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Shape*

stack_name 
�
Cgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
�
Igradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_4/while/LSTM_4/add_1_grad/Shape^gradients/Add_2*
swap_memory( *
T0
�
Hgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Ngradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
Egradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
	elem_type0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Shape_1
�
Egradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Kgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_4/while/LSTM_4/add_1_grad/Shape_1^gradients/Add_2*
T0*
swap_memory( 
�
Jgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0
�
Pgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
+gradients/rnn_4/while/LSTM_4/add_1_grad/SumSumgradients/AddN_5=gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_4/while/LSTM_4/add_1_grad/ReshapeReshape+gradients/rnn_4/while/LSTM_4/add_1_grad/SumHgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_4/while/LSTM_4/add_1_grad/Sum_1Sumgradients/AddN_5?gradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
1gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape_1Reshape-gradients/rnn_4/while/LSTM_4/add_1_grad/Sum_1Jgradients/rnn_4/while/LSTM_4/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape2^gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape_1
�
@gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape9^gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape
�
Bgradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape_19^gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_4/while/LSTM_4/add_1_grad/Reshape_1
i
+gradients/rnn_4/while/LSTM_4/mul_grad/ShapeShapernn_4/while/LSTM_4/Sigmoid*
T0*
out_type0
g
-gradients/rnn_4/while/LSTM_4/mul_grad/Shape_1Shapernn_4/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*>
_class4
20loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Shape*

stack_name 
�
Agradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
�
Ggradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_4/while/LSTM_4/mul_grad/Shape^gradients/Add_2*
T0*
swap_memory( 
�
Fgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Lgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
Cgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Const_1Const*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
Cgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Const_1*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Shape_1*

stack_name *
	elem_type0
�
Cgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Igradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_4/while/LSTM_4/mul_grad/Shape_1^gradients/Add_2*
T0*
swap_memory( 
�
Hgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0
�
Ngradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
)gradients/rnn_4/while/LSTM_4/mul_grad/MulMul@gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependency4gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/ConstConst*
dtype0*)
_class
loc:@rnn_4/while/Identity_3*
valueB :
���������
�
/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/f_accStackV2/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/Const*)
_class
loc:@rnn_4/while/Identity_3*

stack_name *
	elem_type0
�
/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/EnterEnter/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
5gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/Enterrnn_4/while/Identity_3^gradients/Add_2*
T0*
swap_memory( 
�
4gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
:gradients/rnn_4/while/LSTM_4/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_4/while/LSTM_4/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
)gradients/rnn_4/while/LSTM_4/mul_grad/SumSum)gradients/rnn_4/while/LSTM_4/mul_grad/Mul;gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_4/while/LSTM_4/mul_grad/ReshapeReshape)gradients/rnn_4/while/LSTM_4/mul_grad/SumFgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1Mul6gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2@gradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_4/while/LSTM_4/Sigmoid*
valueB :
���������*
dtype0
�
1gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/f_accStackV21gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/Const*-
_class#
!loc:@rnn_4/while/LSTM_4/Sigmoid*

stack_name *
	elem_type0
�
1gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
7gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/Enterrnn_4/while/LSTM_4/Sigmoid^gradients/Add_2*
T0*
swap_memory( 
�
6gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
<gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
+gradients/rnn_4/while/LSTM_4/mul_grad/Sum_1Sum+gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1=gradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_4/while/LSTM_4/mul_grad/Reshape_1Reshape+gradients/rnn_4/while/LSTM_4/mul_grad/Sum_1Hgradients/rnn_4/while/LSTM_4/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_4/while/LSTM_4/mul_grad/tuple/group_depsNoOp.^gradients/rnn_4/while/LSTM_4/mul_grad/Reshape0^gradients/rnn_4/while/LSTM_4/mul_grad/Reshape_1
�
>gradients/rnn_4/while/LSTM_4/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_4/while/LSTM_4/mul_grad/Reshape7^gradients/rnn_4/while/LSTM_4/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Reshape
�
@gradients/rnn_4/while/LSTM_4/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_4/while/LSTM_4/mul_grad/Reshape_17^gradients/rnn_4/while/LSTM_4/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_grad/Reshape_1
m
-gradients/rnn_4/while/LSTM_4/mul_1_grad/ShapeShapernn_4/while/LSTM_4/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape_1Shapernn_4/while/LSTM_4/Tanh*
T0*
out_type0
�
=gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape*
valueB :
���������
�
Cgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Igradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape^gradients/Add_2*
T0*
swap_memory( 
�
Hgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Ngradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
Egradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
	elem_type0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape_1
�
Egradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *)

frame_namernn_4/while/while_context*
T0*
is_constant(
�
Kgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_4/while/LSTM_4/mul_1_grad/Shape_1^gradients/Add_2*
T0*
swap_memory( 
�
Jgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0
�
Pgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
+gradients/rnn_4/while/LSTM_4/mul_1_grad/MulMulBgradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependency_16gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/ConstConst**
_class 
loc:@rnn_4/while/LSTM_4/Tanh*
valueB :
���������*
dtype0
�
1gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/f_accStackV21gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/Const*

stack_name *
	elem_type0**
_class 
loc:@rnn_4/while/LSTM_4/Tanh
�
1gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
7gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/Enterrnn_4/while/LSTM_4/Tanh^gradients/Add_2*
T0*
swap_memory( 
�
6gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
<gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
+gradients/rnn_4/while/LSTM_4/mul_1_grad/SumSum+gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul=gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_4/while/LSTM_4/mul_1_grad/ReshapeReshape+gradients/rnn_4/while/LSTM_4/mul_1_grad/SumHgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1Mul8gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_4/while/LSTM_4/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_4/while/LSTM_4/Sigmoid_1*
valueB :
���������*
dtype0
�
3gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/Const*/
_class%
#!loc:@rnn_4/while/LSTM_4/Sigmoid_1*

stack_name *
	elem_type0
�
3gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
9gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/Enterrnn_4/while/LSTM_4/Sigmoid_1^gradients/Add_2*
swap_memory( *
T0
�
8gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
>gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
-gradients/rnn_4/while/LSTM_4/mul_1_grad/Sum_1Sum-gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1?gradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape_1Reshape-gradients/rnn_4/while/LSTM_4/mul_1_grad/Sum_1Jgradients/rnn_4/while/LSTM_4/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape2^gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape_1
�
@gradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape9^gradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape
�
Bgradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape_19^gradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_4/while/LSTM_4/mul_1_grad/Reshape_1
�
5gradients/rnn_4/while/LSTM_4/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_4/while/LSTM_4/mul_grad/Mul_1/StackPopV2>gradients/rnn_4/while/LSTM_4/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_4/while/LSTM_4/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_4/while/LSTM_4/Tanh_grad/TanhGradTanhGrad6gradients/rnn_4/while/LSTM_4/mul_1_grad/Mul/StackPopV2Bgradients/rnn_4/while/LSTM_4/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_4/while/LSTM_4/add_grad/ShapeShapernn_4/while/LSTM_4/split:2*
T0*
out_type0
h
-gradients/rnn_4/while/LSTM_4/add_grad/Shape_1Const^gradients/Sub_2*
valueB *
dtype0
�
;gradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_4/while/LSTM_4/add_grad/Shape_1*
T0
�
Agradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_4/while/LSTM_4/add_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*>
_class4
20loc:@gradients/rnn_4/while/LSTM_4/add_grad/Shape
�
Agradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
Ggradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_4/while/LSTM_4/add_grad/Shape^gradients/Add_2*
T0*
swap_memory( 
�
Fgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Lgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
)gradients/rnn_4/while/LSTM_4/add_grad/SumSum5gradients/rnn_4/while/LSTM_4/Sigmoid_grad/SigmoidGrad;gradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn_4/while/LSTM_4/add_grad/ReshapeReshape)gradients/rnn_4/while/LSTM_4/add_grad/SumFgradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_4/while/LSTM_4/add_grad/Sum_1Sum5gradients/rnn_4/while/LSTM_4/Sigmoid_grad/SigmoidGrad=gradients/rnn_4/while/LSTM_4/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_4/while/LSTM_4/add_grad/Reshape_1Reshape+gradients/rnn_4/while/LSTM_4/add_grad/Sum_1-gradients/rnn_4/while/LSTM_4/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_4/while/LSTM_4/add_grad/tuple/group_depsNoOp.^gradients/rnn_4/while/LSTM_4/add_grad/Reshape0^gradients/rnn_4/while/LSTM_4/add_grad/Reshape_1
�
>gradients/rnn_4/while/LSTM_4/add_grad/tuple/control_dependencyIdentity-gradients/rnn_4/while/LSTM_4/add_grad/Reshape7^gradients/rnn_4/while/LSTM_4/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_4/while/LSTM_4/add_grad/Reshape
�
@gradients/rnn_4/while/LSTM_4/add_grad/tuple/control_dependency_1Identity/gradients/rnn_4/while/LSTM_4/add_grad/Reshape_17^gradients/rnn_4/while/LSTM_4/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/add_grad/Reshape_1
�
3gradients/rnn_4/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_4/while/LSTM_4/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_4/while/LSTM_4/split_grad/concatConcatV27gradients/rnn_4/while/LSTM_4/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_4/while/LSTM_4/Tanh_grad/TanhGrad>gradients/rnn_4/while/LSTM_4/add_grad/tuple/control_dependency7gradients/rnn_4/while/LSTM_4/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_4/while/LSTM_4/split_grad/concat/Const*
T0*
N*

Tidx0
p
4gradients/rnn_4/while/LSTM_4/split_grad/concat/ConstConst^gradients/Sub_2*
value	B :*
dtype0
�
5gradients/rnn_4/while/LSTM_4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_4/while/LSTM_4/split_grad/concat*
T0*
data_formatNHWC
�
:gradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_4/while/LSTM_4/BiasAdd_grad/BiasAddGrad/^gradients/rnn_4/while/LSTM_4/split_grad/concat
�
Bgradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_4/while/LSTM_4/split_grad/concat;^gradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_4/while/LSTM_4/split_grad/concat
�
Dgradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_4/while/LSTM_4/BiasAdd_grad/BiasAddGrad;^gradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_4/while/LSTM_4/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMulMatMulBgradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/control_dependency5gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
T0
�
5gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul/EnterEnterrnn/LSTM_4/kernel/read*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
1gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1MatMul<gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
7gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/ConstConst*,
_class"
 loc:@rnn_4/while/LSTM_4/concat*
valueB :
���������*
dtype0
�
7gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/Const*,
_class"
 loc:@rnn_4/while/LSTM_4/concat*

stack_name *
	elem_type0
�
7gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
=gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/Enterrnn_4/while/LSTM_4/concat^gradients/Add_2*
T0*
swap_memory( 
�
<gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
Bgradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
9gradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul2^gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1
�
Agradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul:^gradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul
�
Cgradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1:^gradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_4/while/LSTM_4/MatMul_grad/MatMul_1
g
5gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB�*    
�
7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/NextIteration*
N*
T0
�
6gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_2gradients/b_count_10*
T0
�
3gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/AddAdd8gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_4/while/LSTM_4/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/Switch*
T0
j
.gradients/rnn_4/while/LSTM_4/concat_grad/ConstConst^gradients/Sub_2*
value	B :*
dtype0
i
-gradients/rnn_4/while/LSTM_4/concat_grad/RankConst^gradients/Sub_2*
value	B :*
dtype0
�
,gradients/rnn_4/while/LSTM_4/concat_grad/modFloorMod.gradients/rnn_4/while/LSTM_4/concat_grad/Const-gradients/rnn_4/while/LSTM_4/concat_grad/Rank*
T0
o
.gradients/rnn_4/while/LSTM_4/concat_grad/ShapeShapernn_4/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_4/while/LSTM_4/concat_grad/ShapeNShapeN:gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2<gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2_1*
N*
T0*
out_type0
�
5gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/ConstConst*0
_class&
$"loc:@rnn_4/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
5gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_accStackV25gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Const*

stack_name *
	elem_type0*0
_class&
$"loc:@rnn_4/while/TensorArrayReadV3
�
5gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/EnterEnter5gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
;gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Enterrnn_4/while/TensorArrayReadV3^gradients/Add_2*
T0*
swap_memory( 
�
:gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_2*
	elem_type0
�
@gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
7gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Const_1Const*)
_class
loc:@rnn_4/while/Identity_4*
valueB :
���������*
dtype0
�
7gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Const_1*

stack_name *
	elem_type0*)
_class
loc:@rnn_4/while/Identity_4
�
7gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_4/while/while_context
�
=gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/Enter_1rnn_4/while/Identity_4^gradients/Add_2*
T0*
swap_memory( 
�
<gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0
�
Bgradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*
is_constant(
�
5gradients/rnn_4/while/LSTM_4/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_4/while/LSTM_4/concat_grad/mod/gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN1gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN:1*
N
�
.gradients/rnn_4/while/LSTM_4/concat_grad/SliceSliceAgradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/control_dependency5gradients/rnn_4/while/LSTM_4/concat_grad/ConcatOffset/gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_4/while/LSTM_4/concat_grad/Slice_1SliceAgradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/control_dependency7gradients/rnn_4/while/LSTM_4/concat_grad/ConcatOffset:11gradients/rnn_4/while/LSTM_4/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_4/while/LSTM_4/concat_grad/tuple/group_depsNoOp/^gradients/rnn_4/while/LSTM_4/concat_grad/Slice1^gradients/rnn_4/while/LSTM_4/concat_grad/Slice_1
�
Agradients/rnn_4/while/LSTM_4/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_4/while/LSTM_4/concat_grad/Slice:^gradients/rnn_4/while/LSTM_4/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_4/while/LSTM_4/concat_grad/Slice
�
Cgradients/rnn_4/while/LSTM_4/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_4/while/LSTM_4/concat_grad/Slice_1:^gradients/rnn_4/while/LSTM_4/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_4/while/LSTM_4/concat_grad/Slice_1
k
4gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0
�
6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_1<gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/NextIteration*
T0*
N
�
5gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_2gradients/b_count_10*
T0
�
2gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/AddAdd7gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/Switch:1Cgradients/rnn_4/while/LSTM_4/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_2*6
_class,
*(loc:@rnn_4/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_4/TensorArray_1*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context*
T0*6
_class,
*(loc:@rnn_4/while/TensorArrayReadV3/Enter*
is_constant(
�
Vgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6
_class,
*(loc:@rnn_4/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
Jgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_4/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_4/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_4/while/LSTM_4/concat_grad/tuple/control_dependencyJgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_4/while/while_context
�
<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
T0
�
;gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_10*
T0
�
8gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_4/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_4/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_4/while/LSTM_4/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_4/TensorArray_1<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_4/TensorArray_1*
source	gradients
�
mgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_4/TensorArray_1
�
cgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_4/TensorArrayUnstack/rangemgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
element_shape:
�
`gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_4/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_4/transpose_grad/InvertPermutationInvertPermutationrnn_4/concat*
T0
�
(gradients/rnn_4/transpose_grad/transpose	Transposehgradients/rnn_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_4/transpose_grad/InvertPermutation*
Tperm0*
T0
`
2gradients/rnn_3/transpose_1_grad/InvertPermutationInvertPermutationrnn_3/concat_2*
T0
�
*gradients/rnn_3/transpose_1_grad/transpose	Transpose(gradients/rnn_4/transpose_grad/transpose2gradients/rnn_3/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
[gradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_3/TensorArrayrnn_3/while/Exit_2*$
_class
loc:@rnn_3/TensorArray*
source	gradients
�
Wgradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_3/while/Exit_2\^gradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_3/TensorArray
�
agradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_3/TensorArrayStack/range*gradients/rnn_3/transpose_1_grad/transposeWgradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
@
gradients/zeros_like_6	ZerosLikernn_3/while/Exit_3*
T0
@
gradients/zeros_like_7	ZerosLikernn_3/while/Exit_4*
T0
�
(gradients/rnn_3/while/Exit_2_grad/b_exitEnteragradients/rnn_3/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant( 
�
(gradients/rnn_3/while/Exit_3_grad/b_exitEntergradients/zeros_like_6*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
(gradients/rnn_3/while/Exit_4_grad/b_exitEntergradients/zeros_like_7*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant( 
�
,gradients/rnn_3/while/Switch_2_grad/b_switchMerge(gradients/rnn_3/while/Exit_2_grad/b_exit3gradients/rnn_3/while/Switch_2_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_3/while/Switch_3_grad/b_switchMerge(gradients/rnn_3/while/Exit_3_grad/b_exit3gradients/rnn_3/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_3/while/Switch_4_grad/b_switchMerge(gradients/rnn_3/while/Exit_4_grad/b_exit3gradients/rnn_3/while/Switch_4_grad_1/NextIteration*
T0*
N
�
)gradients/rnn_3/while/Merge_2_grad/SwitchSwitch,gradients/rnn_3/while/Switch_2_grad/b_switchgradients/b_count_14*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_2_grad/b_switch
g
3gradients/rnn_3/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_3/while/Merge_2_grad/Switch
�
;gradients/rnn_3/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_3/while/Merge_2_grad/Switch4^gradients/rnn_3/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_2_grad/b_switch
�
=gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_3/while/Merge_2_grad/Switch:14^gradients/rnn_3/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_2_grad/b_switch
�
)gradients/rnn_3/while/Merge_3_grad/SwitchSwitch,gradients/rnn_3/while/Switch_3_grad/b_switchgradients/b_count_14*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_3_grad/b_switch
g
3gradients/rnn_3/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_3/while/Merge_3_grad/Switch
�
;gradients/rnn_3/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_3/while/Merge_3_grad/Switch4^gradients/rnn_3/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_3_grad/b_switch
�
=gradients/rnn_3/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_3/while/Merge_3_grad/Switch:14^gradients/rnn_3/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_3_grad/b_switch
�
)gradients/rnn_3/while/Merge_4_grad/SwitchSwitch,gradients/rnn_3/while/Switch_4_grad/b_switchgradients/b_count_14*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_4_grad/b_switch
g
3gradients/rnn_3/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_3/while/Merge_4_grad/Switch
�
;gradients/rnn_3/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_3/while/Merge_4_grad/Switch4^gradients/rnn_3/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_4_grad/b_switch
�
=gradients/rnn_3/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_3/while/Merge_4_grad/Switch:14^gradients/rnn_3/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_4_grad/b_switch
u
'gradients/rnn_3/while/Enter_2_grad/ExitExit;gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_3/while/Enter_3_grad/ExitExit;gradients/rnn_3/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_3/while/Enter_4_grad/ExitExit;gradients/rnn_3/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_3/while/LSTM_3/mul_2*
source	gradients
�
fgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_3/TensorArray*
T0*+
_class!
loc:@rnn_3/while/LSTM_3/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
\gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_3/while/LSTM_3/mul_2
�
Pgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*)
_class
loc:@rnn_3/while/Identity_1*
valueB :
���������*
dtype0
�
Vgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_3/while/Identity_1
�
Vgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
\gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_3/while/Identity_1^gradients/Add_3*
swap_memory( *
T0
�
[gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
agradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant(
�

Wgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2=^gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV29^gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV29^gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPopV27^gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2\^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_3/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_2_grad/b_switch
�
gradients/AddN_6AddN=gradients/rnn_3/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
N*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_4_grad/b_switch
m
-gradients/rnn_3/while/LSTM_3/mul_2_grad/ShapeShapernn_3/while/LSTM_3/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape_1Shapernn_3/while/LSTM_3/Tanh_1*
T0*
out_type0
�
=gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape*

stack_name 
�
Cgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Igradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape^gradients/Add_3*
T0*
swap_memory( 
�
Hgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Ngradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
Egradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Kgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_3/while/LSTM_3/mul_2_grad/Shape_1^gradients/Add_3*
swap_memory( *
T0
�
Jgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0
�
Pgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/mul_2_grad/MulMulgradients/AddN_66gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/ConstConst*,
_class"
 loc:@rnn_3/while/LSTM_3/Tanh_1*
valueB :
���������*
dtype0
�
1gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/f_accStackV21gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/Const*

stack_name *
	elem_type0*,
_class"
 loc:@rnn_3/while/LSTM_3/Tanh_1
�
1gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
7gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/Enterrnn_3/while/LSTM_3/Tanh_1^gradients/Add_3*
T0*
swap_memory( 
�
6gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
<gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/mul_2_grad/SumSum+gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul=gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_3/while/LSTM_3/mul_2_grad/ReshapeReshape+gradients/rnn_3/while/LSTM_3/mul_2_grad/SumHgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1Mul8gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2gradients/AddN_6*
T0
�
3gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/ConstConst*
dtype0*/
_class%
#!loc:@rnn_3/while/LSTM_3/Sigmoid_2*
valueB :
���������
�
3gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/Const*/
_class%
#!loc:@rnn_3/while/LSTM_3/Sigmoid_2*

stack_name *
	elem_type0
�
3gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
9gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/Enterrnn_3/while/LSTM_3/Sigmoid_2^gradients/Add_3*
swap_memory( *
T0
�
8gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
>gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
-gradients/rnn_3/while/LSTM_3/mul_2_grad/Sum_1Sum-gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1?gradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape_1Reshape-gradients/rnn_3/while/LSTM_3/mul_2_grad/Sum_1Jgradients/rnn_3/while/LSTM_3/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape2^gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape_1
�
@gradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape9^gradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape
�
Bgradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape_19^gradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_3/while/LSTM_3/mul_2_grad/Reshape_1
�
7gradients/rnn_3/while/LSTM_3/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_3/while/LSTM_3/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_3/while/LSTM_3/mul_2_grad/Mul/StackPopV2Bgradients/rnn_3/while/LSTM_3/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_3/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_7AddN=gradients/rnn_3/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_3/while/LSTM_3/Tanh_1_grad/TanhGrad*
T0*?
_class5
31loc:@gradients/rnn_3/while/Switch_3_grad/b_switch*
N
g
-gradients/rnn_3/while/LSTM_3/add_1_grad/ShapeShapernn_3/while/LSTM_3/mul*
T0*
out_type0
k
/gradients/rnn_3/while/LSTM_3/add_1_grad/Shape_1Shapernn_3/while/LSTM_3/mul_1*
T0*
out_type0
�
=gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Shape*

stack_name 
�
Cgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant(
�
Igradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_3/while/LSTM_3/add_1_grad/Shape^gradients/Add_3*
T0*
swap_memory( 
�
Hgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Ngradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
Egradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant(
�
Kgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_3/while/LSTM_3/add_1_grad/Shape_1^gradients/Add_3*
T0*
swap_memory( 
�
Jgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0
�
Pgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/add_1_grad/SumSumgradients/AddN_7=gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_3/while/LSTM_3/add_1_grad/ReshapeReshape+gradients/rnn_3/while/LSTM_3/add_1_grad/SumHgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_3/while/LSTM_3/add_1_grad/Sum_1Sumgradients/AddN_7?gradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape_1Reshape-gradients/rnn_3/while/LSTM_3/add_1_grad/Sum_1Jgradients/rnn_3/while/LSTM_3/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape2^gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape_1
�
@gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape9^gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape
�
Bgradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape_19^gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_3/while/LSTM_3/add_1_grad/Reshape_1
i
+gradients/rnn_3/while/LSTM_3/mul_grad/ShapeShapernn_3/while/LSTM_3/Sigmoid*
T0*
out_type0
g
-gradients/rnn_3/while/LSTM_3/mul_grad/Shape_1Shapernn_3/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/ConstConst*
dtype0*>
_class4
20loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Shape*
valueB :
���������
�
Agradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Ggradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_3/while/LSTM_3/mul_grad/Shape^gradients/Add_3*
T0*
swap_memory( 
�
Fgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Lgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant(
�
Cgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Const_1Const*
dtype0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Shape_1*
valueB :
���������
�
Cgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Const_1*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Shape_1*

stack_name *
	elem_type0
�
Cgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Igradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_3/while/LSTM_3/mul_grad/Shape_1^gradients/Add_3*
T0*
swap_memory( 
�
Hgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0
�
Ngradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant(
�
)gradients/rnn_3/while/LSTM_3/mul_grad/MulMul@gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependency4gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/ConstConst*)
_class
loc:@rnn_3/while/Identity_3*
valueB :
���������*
dtype0
�
/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/f_accStackV2/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_3/while/Identity_3
�
/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/EnterEnter/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
5gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/Enterrnn_3/while/Identity_3^gradients/Add_3*
T0*
swap_memory( 
�
4gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
:gradients/rnn_3/while/LSTM_3/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_3/while/LSTM_3/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
)gradients/rnn_3/while/LSTM_3/mul_grad/SumSum)gradients/rnn_3/while/LSTM_3/mul_grad/Mul;gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_3/while/LSTM_3/mul_grad/ReshapeReshape)gradients/rnn_3/while/LSTM_3/mul_grad/SumFgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1Mul6gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2@gradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_3/while/LSTM_3/Sigmoid*
valueB :
���������*
dtype0
�
1gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/f_accStackV21gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/Const*-
_class#
!loc:@rnn_3/while/LSTM_3/Sigmoid*

stack_name *
	elem_type0
�
1gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/f_acc*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant(
�
7gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/Enterrnn_3/while/LSTM_3/Sigmoid^gradients/Add_3*
T0*
swap_memory( 
�
6gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
<gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/mul_grad/Sum_1Sum+gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1=gradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_3/while/LSTM_3/mul_grad/Reshape_1Reshape+gradients/rnn_3/while/LSTM_3/mul_grad/Sum_1Hgradients/rnn_3/while/LSTM_3/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_3/while/LSTM_3/mul_grad/tuple/group_depsNoOp.^gradients/rnn_3/while/LSTM_3/mul_grad/Reshape0^gradients/rnn_3/while/LSTM_3/mul_grad/Reshape_1
�
>gradients/rnn_3/while/LSTM_3/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_3/while/LSTM_3/mul_grad/Reshape7^gradients/rnn_3/while/LSTM_3/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Reshape
�
@gradients/rnn_3/while/LSTM_3/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_3/while/LSTM_3/mul_grad/Reshape_17^gradients/rnn_3/while/LSTM_3/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_grad/Reshape_1
m
-gradients/rnn_3/while/LSTM_3/mul_1_grad/ShapeShapernn_3/while/LSTM_3/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape_1Shapernn_3/while/LSTM_3/Tanh*
T0*
out_type0
�
=gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape
�
Cgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Igradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape^gradients/Add_3*
T0*
swap_memory( 
�
Hgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Ngradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
Egradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Kgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_3/while/LSTM_3/mul_1_grad/Shape_1^gradients/Add_3*
T0*
swap_memory( 
�
Jgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0
�
Pgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/mul_1_grad/MulMulBgradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependency_16gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/ConstConst**
_class 
loc:@rnn_3/while/LSTM_3/Tanh*
valueB :
���������*
dtype0
�
1gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/f_accStackV21gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/Const**
_class 
loc:@rnn_3/while/LSTM_3/Tanh*

stack_name *
	elem_type0
�
1gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
7gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/Enterrnn_3/while/LSTM_3/Tanh^gradients/Add_3*
T0*
swap_memory( 
�
6gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
<gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
+gradients/rnn_3/while/LSTM_3/mul_1_grad/SumSum+gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul=gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_3/while/LSTM_3/mul_1_grad/ReshapeReshape+gradients/rnn_3/while/LSTM_3/mul_1_grad/SumHgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1Mul8gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_3/while/LSTM_3/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_3/while/LSTM_3/Sigmoid_1*
valueB :
���������*
dtype0
�
3gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/Const*
	elem_type0*/
_class%
#!loc:@rnn_3/while/LSTM_3/Sigmoid_1*

stack_name 
�
3gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/f_acc*
parallel_iterations *)

frame_namernn_3/while/while_context*
T0*
is_constant(
�
9gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/Enterrnn_3/while/LSTM_3/Sigmoid_1^gradients/Add_3*
T0*
swap_memory( 
�
8gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
>gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
-gradients/rnn_3/while/LSTM_3/mul_1_grad/Sum_1Sum-gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1?gradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape_1Reshape-gradients/rnn_3/while/LSTM_3/mul_1_grad/Sum_1Jgradients/rnn_3/while/LSTM_3/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape2^gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape_1
�
@gradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape9^gradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape
�
Bgradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape_19^gradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_3/while/LSTM_3/mul_1_grad/Reshape_1
�
5gradients/rnn_3/while/LSTM_3/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_3/while/LSTM_3/mul_grad/Mul_1/StackPopV2>gradients/rnn_3/while/LSTM_3/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_3/while/LSTM_3/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_3/while/LSTM_3/Tanh_grad/TanhGradTanhGrad6gradients/rnn_3/while/LSTM_3/mul_1_grad/Mul/StackPopV2Bgradients/rnn_3/while/LSTM_3/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_3/while/LSTM_3/add_grad/ShapeShapernn_3/while/LSTM_3/split:2*
T0*
out_type0
h
-gradients/rnn_3/while/LSTM_3/add_grad/Shape_1Const^gradients/Sub_3*
valueB *
dtype0
�
;gradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_3/while/LSTM_3/add_grad/Shape_1*
T0
�
Agradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_3/while/LSTM_3/add_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/Const*
	elem_type0*>
_class4
20loc:@gradients/rnn_3/while/LSTM_3/add_grad/Shape*

stack_name 
�
Agradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
Ggradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_3/while/LSTM_3/add_grad/Shape^gradients/Add_3*
T0*
swap_memory( 
�
Fgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Lgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
)gradients/rnn_3/while/LSTM_3/add_grad/SumSum5gradients/rnn_3/while/LSTM_3/Sigmoid_grad/SigmoidGrad;gradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn_3/while/LSTM_3/add_grad/ReshapeReshape)gradients/rnn_3/while/LSTM_3/add_grad/SumFgradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_3/while/LSTM_3/add_grad/Sum_1Sum5gradients/rnn_3/while/LSTM_3/Sigmoid_grad/SigmoidGrad=gradients/rnn_3/while/LSTM_3/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_3/while/LSTM_3/add_grad/Reshape_1Reshape+gradients/rnn_3/while/LSTM_3/add_grad/Sum_1-gradients/rnn_3/while/LSTM_3/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_3/while/LSTM_3/add_grad/tuple/group_depsNoOp.^gradients/rnn_3/while/LSTM_3/add_grad/Reshape0^gradients/rnn_3/while/LSTM_3/add_grad/Reshape_1
�
>gradients/rnn_3/while/LSTM_3/add_grad/tuple/control_dependencyIdentity-gradients/rnn_3/while/LSTM_3/add_grad/Reshape7^gradients/rnn_3/while/LSTM_3/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_3/while/LSTM_3/add_grad/Reshape
�
@gradients/rnn_3/while/LSTM_3/add_grad/tuple/control_dependency_1Identity/gradients/rnn_3/while/LSTM_3/add_grad/Reshape_17^gradients/rnn_3/while/LSTM_3/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/add_grad/Reshape_1
�
3gradients/rnn_3/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_3/while/LSTM_3/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_3/while/LSTM_3/split_grad/concatConcatV27gradients/rnn_3/while/LSTM_3/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_3/while/LSTM_3/Tanh_grad/TanhGrad>gradients/rnn_3/while/LSTM_3/add_grad/tuple/control_dependency7gradients/rnn_3/while/LSTM_3/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_3/while/LSTM_3/split_grad/concat/Const*
N*

Tidx0*
T0
p
4gradients/rnn_3/while/LSTM_3/split_grad/concat/ConstConst^gradients/Sub_3*
value	B :*
dtype0
�
5gradients/rnn_3/while/LSTM_3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_3/while/LSTM_3/split_grad/concat*
data_formatNHWC*
T0
�
:gradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_3/while/LSTM_3/BiasAdd_grad/BiasAddGrad/^gradients/rnn_3/while/LSTM_3/split_grad/concat
�
Bgradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_3/while/LSTM_3/split_grad/concat;^gradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_3/while/LSTM_3/split_grad/concat
�
Dgradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_3/while/LSTM_3/BiasAdd_grad/BiasAddGrad;^gradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_3/while/LSTM_3/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMulMatMulBgradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/control_dependency5gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *
transpose_b(
�
5gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul/EnterEnterrnn/LSTM_3/kernel/read*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
1gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1MatMul<gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
7gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/ConstConst*
dtype0*,
_class"
 loc:@rnn_3/while/LSTM_3/concat*
valueB :
���������
�
7gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/Const*

stack_name *
	elem_type0*,
_class"
 loc:@rnn_3/while/LSTM_3/concat
�
7gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
=gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/Enterrnn_3/while/LSTM_3/concat^gradients/Add_3*
T0*
swap_memory( 
�
<gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
Bgradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant(
�
9gradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul2^gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1
�
Agradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul:^gradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul
�
Cgradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1:^gradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_3/while/LSTM_3/MatMul_grad/MatMul_1
g
5gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB�*    
�
7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/NextIteration*
N*
T0
�
6gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_2gradients/b_count_14*
T0
�
3gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/AddAdd8gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_3/while/LSTM_3/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/Switch*
T0
j
.gradients/rnn_3/while/LSTM_3/concat_grad/ConstConst^gradients/Sub_3*
value	B :*
dtype0
i
-gradients/rnn_3/while/LSTM_3/concat_grad/RankConst^gradients/Sub_3*
value	B :*
dtype0
�
,gradients/rnn_3/while/LSTM_3/concat_grad/modFloorMod.gradients/rnn_3/while/LSTM_3/concat_grad/Const-gradients/rnn_3/while/LSTM_3/concat_grad/Rank*
T0
o
.gradients/rnn_3/while/LSTM_3/concat_grad/ShapeShapernn_3/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_3/while/LSTM_3/concat_grad/ShapeNShapeN:gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2<gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N
�
5gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/ConstConst*
dtype0*0
_class&
$"loc:@rnn_3/while/TensorArrayReadV3*
valueB :
���������
�
5gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_accStackV25gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Const*0
_class&
$"loc:@rnn_3/while/TensorArrayReadV3*

stack_name *
	elem_type0
�
5gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/EnterEnter5gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
;gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Enterrnn_3/while/TensorArrayReadV3^gradients/Add_3*
T0*
swap_memory( 
�
:gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_3*
	elem_type0
�
@gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
7gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Const_1Const*)
_class
loc:@rnn_3/while/Identity_4*
valueB :
���������*
dtype0
�
7gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Const_1*)
_class
loc:@rnn_3/while/Identity_4*

stack_name *
	elem_type0
�
7gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_3/while/while_context
�
=gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/Enter_1rnn_3/while/Identity_4^gradients/Add_3*
T0*
swap_memory( 
�
<gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_3*
	elem_type0
�
Bgradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
5gradients/rnn_3/while/LSTM_3/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_3/while/LSTM_3/concat_grad/mod/gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN1gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN:1*
N
�
.gradients/rnn_3/while/LSTM_3/concat_grad/SliceSliceAgradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/control_dependency5gradients/rnn_3/while/LSTM_3/concat_grad/ConcatOffset/gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_3/while/LSTM_3/concat_grad/Slice_1SliceAgradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/control_dependency7gradients/rnn_3/while/LSTM_3/concat_grad/ConcatOffset:11gradients/rnn_3/while/LSTM_3/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_3/while/LSTM_3/concat_grad/tuple/group_depsNoOp/^gradients/rnn_3/while/LSTM_3/concat_grad/Slice1^gradients/rnn_3/while/LSTM_3/concat_grad/Slice_1
�
Agradients/rnn_3/while/LSTM_3/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_3/while/LSTM_3/concat_grad/Slice:^gradients/rnn_3/while/LSTM_3/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_3/while/LSTM_3/concat_grad/Slice
�
Cgradients/rnn_3/while/LSTM_3/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_3/while/LSTM_3/concat_grad/Slice_1:^gradients/rnn_3/while/LSTM_3/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_3/while/LSTM_3/concat_grad/Slice_1
k
4gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_1<gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/NextIteration*
N*
T0
�
5gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_2gradients/b_count_14*
T0
�
2gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/AddAdd7gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/Switch:1Cgradients/rnn_3/while/LSTM_3/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_3*6
_class,
*(loc:@rnn_3/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_3/TensorArray_1*
T0*6
_class,
*(loc:@rnn_3/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context
�
Vgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*6
_class,
*(loc:@rnn_3/while/TensorArrayReadV3/Enter*
is_constant(
�
Jgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_3/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_3/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_3/while/LSTM_3/concat_grad/tuple/control_dependencyJgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *3

frame_name%#gradients/rnn_3/while/while_context*
T0*
is_constant( 
�
<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
;gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_14*
T0
�
8gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_3/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_3/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_3/while/LSTM_3/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_3/TensorArray_1<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_3/TensorArray_1*
source	gradients
�
mgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_3/TensorArray_1
�
cgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_3/TensorArrayUnstack/rangemgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
element_shape:
�
`gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_3/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_3/transpose_grad/InvertPermutationInvertPermutationrnn_3/concat*
T0
�
(gradients/rnn_3/transpose_grad/transpose	Transposehgradients/rnn_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_3/transpose_grad/InvertPermutation*
T0*
Tperm0
`
2gradients/rnn_2/transpose_1_grad/InvertPermutationInvertPermutationrnn_2/concat_2*
T0
�
*gradients/rnn_2/transpose_1_grad/transpose	Transpose(gradients/rnn_3/transpose_grad/transpose2gradients/rnn_2/transpose_1_grad/InvertPermutation*
T0*
Tperm0
�
[gradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_2/TensorArrayrnn_2/while/Exit_2*$
_class
loc:@rnn_2/TensorArray*
source	gradients
�
Wgradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_2/while/Exit_2\^gradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_2/TensorArray
�
agradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_2/TensorArrayStack/range*gradients/rnn_2/transpose_1_grad/transposeWgradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
@
gradients/zeros_like_8	ZerosLikernn_2/while/Exit_3*
T0
@
gradients/zeros_like_9	ZerosLikernn_2/while/Exit_4*
T0
�
(gradients/rnn_2/while/Exit_2_grad/b_exitEnteragradients/rnn_2/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
(gradients/rnn_2/while/Exit_3_grad/b_exitEntergradients/zeros_like_8*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant( 
�
(gradients/rnn_2/while/Exit_4_grad/b_exitEntergradients/zeros_like_9*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
,gradients/rnn_2/while/Switch_2_grad/b_switchMerge(gradients/rnn_2/while/Exit_2_grad/b_exit3gradients/rnn_2/while/Switch_2_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_2/while/Switch_3_grad/b_switchMerge(gradients/rnn_2/while/Exit_3_grad/b_exit3gradients/rnn_2/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_2/while/Switch_4_grad/b_switchMerge(gradients/rnn_2/while/Exit_4_grad/b_exit3gradients/rnn_2/while/Switch_4_grad_1/NextIteration*
T0*
N
�
)gradients/rnn_2/while/Merge_2_grad/SwitchSwitch,gradients/rnn_2/while/Switch_2_grad/b_switchgradients/b_count_18*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_2_grad/b_switch
g
3gradients/rnn_2/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_2/while/Merge_2_grad/Switch
�
;gradients/rnn_2/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_2/while/Merge_2_grad/Switch4^gradients/rnn_2/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_2_grad/b_switch
�
=gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_2/while/Merge_2_grad/Switch:14^gradients/rnn_2/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_2_grad/b_switch
�
)gradients/rnn_2/while/Merge_3_grad/SwitchSwitch,gradients/rnn_2/while/Switch_3_grad/b_switchgradients/b_count_18*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_3_grad/b_switch
g
3gradients/rnn_2/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_2/while/Merge_3_grad/Switch
�
;gradients/rnn_2/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_2/while/Merge_3_grad/Switch4^gradients/rnn_2/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_3_grad/b_switch
�
=gradients/rnn_2/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_2/while/Merge_3_grad/Switch:14^gradients/rnn_2/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_3_grad/b_switch
�
)gradients/rnn_2/while/Merge_4_grad/SwitchSwitch,gradients/rnn_2/while/Switch_4_grad/b_switchgradients/b_count_18*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_4_grad/b_switch
g
3gradients/rnn_2/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_2/while/Merge_4_grad/Switch
�
;gradients/rnn_2/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_2/while/Merge_4_grad/Switch4^gradients/rnn_2/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_4_grad/b_switch
�
=gradients/rnn_2/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_2/while/Merge_4_grad/Switch:14^gradients/rnn_2/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_4_grad/b_switch
u
'gradients/rnn_2/while/Enter_2_grad/ExitExit;gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_2/while/Enter_3_grad/ExitExit;gradients/rnn_2/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_2/while/Enter_4_grad/ExitExit;gradients/rnn_2/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_2/while/LSTM_2/mul_2*
source	gradients
�
fgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_2/TensorArray*
T0*+
_class!
loc:@rnn_2/while/LSTM_2/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
\gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_2/while/LSTM_2/mul_2
�
Pgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*)
_class
loc:@rnn_2/while/Identity_1*
valueB :
���������*
dtype0
�
Vgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*)
_class
loc:@rnn_2/while/Identity_1*

stack_name 
�
Vgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
\gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_2/while/Identity_1^gradients/Add_4*
T0*
swap_memory( 
�
[gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
agradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�

Wgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2=^gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV29^gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV29^gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPopV27^gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2\^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_2/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_2_grad/b_switch
�
gradients/AddN_8AddN=gradients/rnn_2/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_4_grad/b_switch*
N
m
-gradients/rnn_2/while/LSTM_2/mul_2_grad/ShapeShapernn_2/while/LSTM_2/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape_1Shapernn_2/while/LSTM_2/Tanh_1*
T0*
out_type0
�
=gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape*
valueB :
���������
�
Cgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape
�
Cgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Igradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape^gradients/Add_4*
T0*
swap_memory( 
�
Hgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Ngradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
Egradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Const_1*

stack_name *
	elem_type0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape_1
�
Egradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Kgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_2/while/LSTM_2/mul_2_grad/Shape_1^gradients/Add_4*
T0*
swap_memory( 
�
Jgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_4*
	elem_type0
�
Pgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
+gradients/rnn_2/while/LSTM_2/mul_2_grad/MulMulgradients/AddN_86gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/ConstConst*,
_class"
 loc:@rnn_2/while/LSTM_2/Tanh_1*
valueB :
���������*
dtype0
�
1gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/f_accStackV21gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/Const*
	elem_type0*,
_class"
 loc:@rnn_2/while/LSTM_2/Tanh_1*

stack_name 
�
1gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
7gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/Enterrnn_2/while/LSTM_2/Tanh_1^gradients/Add_4*
T0*
swap_memory( 
�
6gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
<gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
+gradients/rnn_2/while/LSTM_2/mul_2_grad/SumSum+gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul=gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_2/while/LSTM_2/mul_2_grad/ReshapeReshape+gradients/rnn_2/while/LSTM_2/mul_2_grad/SumHgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1Mul8gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2gradients/AddN_8*
T0
�
3gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/ConstConst*
dtype0*/
_class%
#!loc:@rnn_2/while/LSTM_2/Sigmoid_2*
valueB :
���������
�
3gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/Const*
	elem_type0*/
_class%
#!loc:@rnn_2/while/LSTM_2/Sigmoid_2*

stack_name 
�
3gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
9gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/Enterrnn_2/while/LSTM_2/Sigmoid_2^gradients/Add_4*
T0*
swap_memory( 
�
8gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
>gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
-gradients/rnn_2/while/LSTM_2/mul_2_grad/Sum_1Sum-gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1?gradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape_1Reshape-gradients/rnn_2/while/LSTM_2/mul_2_grad/Sum_1Jgradients/rnn_2/while/LSTM_2/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape2^gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape_1
�
@gradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape9^gradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape
�
Bgradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape_19^gradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_2/while/LSTM_2/mul_2_grad/Reshape_1
�
7gradients/rnn_2/while/LSTM_2/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_2/while/LSTM_2/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_2/while/LSTM_2/mul_2_grad/Mul/StackPopV2Bgradients/rnn_2/while/LSTM_2/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_2/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_9AddN=gradients/rnn_2/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_2/while/LSTM_2/Tanh_1_grad/TanhGrad*
T0*?
_class5
31loc:@gradients/rnn_2/while/Switch_3_grad/b_switch*
N
g
-gradients/rnn_2/while/LSTM_2/add_1_grad/ShapeShapernn_2/while/LSTM_2/mul*
T0*
out_type0
k
/gradients/rnn_2/while/LSTM_2/add_1_grad/Shape_1Shapernn_2/while/LSTM_2/mul_1*
T0*
out_type0
�
=gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Shape*

stack_name 
�
Cgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Igradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_2/while/LSTM_2/add_1_grad/Shape^gradients/Add_4*
T0*
swap_memory( 
�
Hgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Ngradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
Egradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Kgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_2/while/LSTM_2/add_1_grad/Shape_1^gradients/Add_4*
T0*
swap_memory( 
�
Jgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_4*
	elem_type0
�
Pgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
+gradients/rnn_2/while/LSTM_2/add_1_grad/SumSumgradients/AddN_9=gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_2/while/LSTM_2/add_1_grad/ReshapeReshape+gradients/rnn_2/while/LSTM_2/add_1_grad/SumHgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_2/while/LSTM_2/add_1_grad/Sum_1Sumgradients/AddN_9?gradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape_1Reshape-gradients/rnn_2/while/LSTM_2/add_1_grad/Sum_1Jgradients/rnn_2/while/LSTM_2/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape2^gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape_1
�
@gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape9^gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape
�
Bgradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape_19^gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_2/while/LSTM_2/add_1_grad/Reshape_1
i
+gradients/rnn_2/while/LSTM_2/mul_grad/ShapeShapernn_2/while/LSTM_2/Sigmoid*
T0*
out_type0
g
-gradients/rnn_2/while/LSTM_2/mul_grad/Shape_1Shapernn_2/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/ConstConst*
dtype0*>
_class4
20loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Shape*
valueB :
���������
�
Agradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Ggradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_2/while/LSTM_2/mul_grad/Shape^gradients/Add_4*
T0*
swap_memory( 
�
Fgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Lgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
Cgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Const_1Const*
dtype0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Shape_1*
valueB :
���������
�
Cgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Shape_1*

stack_name 
�
Cgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Igradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_2/while/LSTM_2/mul_grad/Shape_1^gradients/Add_4*
T0*
swap_memory( 
�
Hgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_4*
	elem_type0
�
Ngradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
)gradients/rnn_2/while/LSTM_2/mul_grad/MulMul@gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependency4gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/ConstConst*)
_class
loc:@rnn_2/while/Identity_3*
valueB :
���������*
dtype0
�
/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/f_accStackV2/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/Const*)
_class
loc:@rnn_2/while/Identity_3*

stack_name *
	elem_type0
�
/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/EnterEnter/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
5gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/Enterrnn_2/while/Identity_3^gradients/Add_4*
T0*
swap_memory( 
�
4gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
:gradients/rnn_2/while/LSTM_2/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_2/while/LSTM_2/mul_grad/Mul/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
)gradients/rnn_2/while/LSTM_2/mul_grad/SumSum)gradients/rnn_2/while/LSTM_2/mul_grad/Mul;gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_2/while/LSTM_2/mul_grad/ReshapeReshape)gradients/rnn_2/while/LSTM_2/mul_grad/SumFgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1Mul6gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2@gradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_2/while/LSTM_2/Sigmoid*
valueB :
���������*
dtype0
�
1gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/f_accStackV21gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/Const*

stack_name *
	elem_type0*-
_class#
!loc:@rnn_2/while/LSTM_2/Sigmoid
�
1gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
7gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/Enterrnn_2/while/LSTM_2/Sigmoid^gradients/Add_4*
T0*
swap_memory( 
�
6gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
<gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
+gradients/rnn_2/while/LSTM_2/mul_grad/Sum_1Sum+gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1=gradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_2/while/LSTM_2/mul_grad/Reshape_1Reshape+gradients/rnn_2/while/LSTM_2/mul_grad/Sum_1Hgradients/rnn_2/while/LSTM_2/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_2/while/LSTM_2/mul_grad/tuple/group_depsNoOp.^gradients/rnn_2/while/LSTM_2/mul_grad/Reshape0^gradients/rnn_2/while/LSTM_2/mul_grad/Reshape_1
�
>gradients/rnn_2/while/LSTM_2/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_2/while/LSTM_2/mul_grad/Reshape7^gradients/rnn_2/while/LSTM_2/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Reshape
�
@gradients/rnn_2/while/LSTM_2/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_2/while/LSTM_2/mul_grad/Reshape_17^gradients/rnn_2/while/LSTM_2/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_grad/Reshape_1
m
-gradients/rnn_2/while/LSTM_2/mul_1_grad/ShapeShapernn_2/while/LSTM_2/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape_1Shapernn_2/while/LSTM_2/Tanh*
T0*
out_type0
�
=gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape*
valueB :
���������
�
Cgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape*

stack_name 
�
Cgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Igradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape^gradients/Add_4*
T0*
swap_memory( 
�
Hgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Ngradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
Egradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape_1*

stack_name 
�
Egradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *)

frame_namernn_2/while/while_context*
T0*
is_constant(
�
Kgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_2/while/LSTM_2/mul_1_grad/Shape_1^gradients/Add_4*
swap_memory( *
T0
�
Jgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_4*
	elem_type0
�
Pgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
+gradients/rnn_2/while/LSTM_2/mul_1_grad/MulMulBgradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependency_16gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/ConstConst**
_class 
loc:@rnn_2/while/LSTM_2/Tanh*
valueB :
���������*
dtype0
�
1gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/f_accStackV21gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/Const*

stack_name *
	elem_type0**
_class 
loc:@rnn_2/while/LSTM_2/Tanh
�
1gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
7gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/Enterrnn_2/while/LSTM_2/Tanh^gradients/Add_4*
T0*
swap_memory( 
�
6gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
<gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
+gradients/rnn_2/while/LSTM_2/mul_1_grad/SumSum+gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul=gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_2/while/LSTM_2/mul_1_grad/ReshapeReshape+gradients/rnn_2/while/LSTM_2/mul_1_grad/SumHgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1Mul8gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_2/while/LSTM_2/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/ConstConst*/
_class%
#!loc:@rnn_2/while/LSTM_2/Sigmoid_1*
valueB :
���������*
dtype0
�
3gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/Const*/
_class%
#!loc:@rnn_2/while/LSTM_2/Sigmoid_1*

stack_name *
	elem_type0
�
3gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
9gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/Enterrnn_2/while/LSTM_2/Sigmoid_1^gradients/Add_4*
T0*
swap_memory( 
�
8gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
>gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
-gradients/rnn_2/while/LSTM_2/mul_1_grad/Sum_1Sum-gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1?gradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
1gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape_1Reshape-gradients/rnn_2/while/LSTM_2/mul_1_grad/Sum_1Jgradients/rnn_2/while/LSTM_2/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape2^gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape_1
�
@gradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape9^gradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape
�
Bgradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape_19^gradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_2/while/LSTM_2/mul_1_grad/Reshape_1
�
5gradients/rnn_2/while/LSTM_2/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_2/while/LSTM_2/mul_grad/Mul_1/StackPopV2>gradients/rnn_2/while/LSTM_2/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_2/while/LSTM_2/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_2/while/LSTM_2/Tanh_grad/TanhGradTanhGrad6gradients/rnn_2/while/LSTM_2/mul_1_grad/Mul/StackPopV2Bgradients/rnn_2/while/LSTM_2/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_2/while/LSTM_2/add_grad/ShapeShapernn_2/while/LSTM_2/split:2*
T0*
out_type0
h
-gradients/rnn_2/while/LSTM_2/add_grad/Shape_1Const^gradients/Sub_4*
valueB *
dtype0
�
;gradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_2/while/LSTM_2/add_grad/Shape_1*
T0
�
Agradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_2/while/LSTM_2/add_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_2/while/LSTM_2/add_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
Ggradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_2/while/LSTM_2/add_grad/Shape^gradients/Add_4*
T0*
swap_memory( 
�
Fgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Lgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
)gradients/rnn_2/while/LSTM_2/add_grad/SumSum5gradients/rnn_2/while/LSTM_2/Sigmoid_grad/SigmoidGrad;gradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_2/while/LSTM_2/add_grad/ReshapeReshape)gradients/rnn_2/while/LSTM_2/add_grad/SumFgradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_2/while/LSTM_2/add_grad/Sum_1Sum5gradients/rnn_2/while/LSTM_2/Sigmoid_grad/SigmoidGrad=gradients/rnn_2/while/LSTM_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_2/while/LSTM_2/add_grad/Reshape_1Reshape+gradients/rnn_2/while/LSTM_2/add_grad/Sum_1-gradients/rnn_2/while/LSTM_2/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_2/while/LSTM_2/add_grad/tuple/group_depsNoOp.^gradients/rnn_2/while/LSTM_2/add_grad/Reshape0^gradients/rnn_2/while/LSTM_2/add_grad/Reshape_1
�
>gradients/rnn_2/while/LSTM_2/add_grad/tuple/control_dependencyIdentity-gradients/rnn_2/while/LSTM_2/add_grad/Reshape7^gradients/rnn_2/while/LSTM_2/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_2/while/LSTM_2/add_grad/Reshape
�
@gradients/rnn_2/while/LSTM_2/add_grad/tuple/control_dependency_1Identity/gradients/rnn_2/while/LSTM_2/add_grad/Reshape_17^gradients/rnn_2/while/LSTM_2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/add_grad/Reshape_1
�
3gradients/rnn_2/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_2/while/LSTM_2/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_2/while/LSTM_2/split_grad/concatConcatV27gradients/rnn_2/while/LSTM_2/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_2/while/LSTM_2/Tanh_grad/TanhGrad>gradients/rnn_2/while/LSTM_2/add_grad/tuple/control_dependency7gradients/rnn_2/while/LSTM_2/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_2/while/LSTM_2/split_grad/concat/Const*
T0*
N*

Tidx0
p
4gradients/rnn_2/while/LSTM_2/split_grad/concat/ConstConst^gradients/Sub_4*
value	B :*
dtype0
�
5gradients/rnn_2/while/LSTM_2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_2/while/LSTM_2/split_grad/concat*
T0*
data_formatNHWC
�
:gradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_2/while/LSTM_2/BiasAdd_grad/BiasAddGrad/^gradients/rnn_2/while/LSTM_2/split_grad/concat
�
Bgradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_2/while/LSTM_2/split_grad/concat;^gradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_2/while/LSTM_2/split_grad/concat
�
Dgradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_2/while/LSTM_2/BiasAdd_grad/BiasAddGrad;^gradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_2/while/LSTM_2/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMulMatMulBgradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/control_dependency5gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *
transpose_b(
�
5gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul/EnterEnterrnn/LSTM_2/kernel/read*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
1gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1MatMul<gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
7gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/ConstConst*,
_class"
 loc:@rnn_2/while/LSTM_2/concat*
valueB :
���������*
dtype0
�
7gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/Const*,
_class"
 loc:@rnn_2/while/LSTM_2/concat*

stack_name *
	elem_type0
�
7gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
=gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/Enterrnn_2/while/LSTM_2/concat^gradients/Add_4*
swap_memory( *
T0
�
<gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
Bgradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
9gradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul2^gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1
�
Agradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul:^gradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul
�
Cgradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1:^gradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_2/while/LSTM_2/MatMul_grad/MatMul_1
g
5gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
6gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_2gradients/b_count_18*
T0
�
3gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/AddAdd8gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_2/while/LSTM_2/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/Switch*
T0
j
.gradients/rnn_2/while/LSTM_2/concat_grad/ConstConst^gradients/Sub_4*
value	B :*
dtype0
i
-gradients/rnn_2/while/LSTM_2/concat_grad/RankConst^gradients/Sub_4*
value	B :*
dtype0
�
,gradients/rnn_2/while/LSTM_2/concat_grad/modFloorMod.gradients/rnn_2/while/LSTM_2/concat_grad/Const-gradients/rnn_2/while/LSTM_2/concat_grad/Rank*
T0
o
.gradients/rnn_2/while/LSTM_2/concat_grad/ShapeShapernn_2/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_2/while/LSTM_2/concat_grad/ShapeNShapeN:gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2<gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2_1*
N*
T0*
out_type0
�
5gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/ConstConst*0
_class&
$"loc:@rnn_2/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
5gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_accStackV25gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Const*0
_class&
$"loc:@rnn_2/while/TensorArrayReadV3*

stack_name *
	elem_type0
�
5gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/EnterEnter5gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_acc*
parallel_iterations *)

frame_namernn_2/while/while_context*
T0*
is_constant(
�
;gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Enterrnn_2/while/TensorArrayReadV3^gradients/Add_4*
swap_memory( *
T0
�
:gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_4*
	elem_type0
�
@gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
7gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Const_1Const*)
_class
loc:@rnn_2/while/Identity_4*
valueB :
���������*
dtype0
�
7gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Const_1*)
_class
loc:@rnn_2/while/Identity_4*

stack_name *
	elem_type0
�
7gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_2/while/while_context
�
=gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/Enter_1rnn_2/while/Identity_4^gradients/Add_4*
T0*
swap_memory( 
�
<gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_4*
	elem_type0
�
Bgradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN/f_acc_1*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*
is_constant(
�
5gradients/rnn_2/while/LSTM_2/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_2/while/LSTM_2/concat_grad/mod/gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN1gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN:1*
N
�
.gradients/rnn_2/while/LSTM_2/concat_grad/SliceSliceAgradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/control_dependency5gradients/rnn_2/while/LSTM_2/concat_grad/ConcatOffset/gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_2/while/LSTM_2/concat_grad/Slice_1SliceAgradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/control_dependency7gradients/rnn_2/while/LSTM_2/concat_grad/ConcatOffset:11gradients/rnn_2/while/LSTM_2/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_2/while/LSTM_2/concat_grad/tuple/group_depsNoOp/^gradients/rnn_2/while/LSTM_2/concat_grad/Slice1^gradients/rnn_2/while/LSTM_2/concat_grad/Slice_1
�
Agradients/rnn_2/while/LSTM_2/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_2/while/LSTM_2/concat_grad/Slice:^gradients/rnn_2/while/LSTM_2/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_2/while/LSTM_2/concat_grad/Slice
�
Cgradients/rnn_2/while/LSTM_2/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_2/while/LSTM_2/concat_grad/Slice_1:^gradients/rnn_2/while/LSTM_2/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_2/while/LSTM_2/concat_grad/Slice_1
k
4gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0
�
6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_1<gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/NextIteration*
T0*
N
�
5gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_2gradients/b_count_18*
T0
�
2gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/AddAdd7gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/Switch:1Cgradients/rnn_2/while/LSTM_2/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_4*6
_class,
*(loc:@rnn_2/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_2/TensorArray_1*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context*
T0*6
_class,
*(loc:@rnn_2/while/TensorArrayReadV3/Enter*
is_constant(
�
Vgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6
_class,
*(loc:@rnn_2/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
Jgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_2/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_2/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_2/while/LSTM_2/concat_grad/tuple/control_dependencyJgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_2/while/while_context
�
<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
T0
�
;gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_18*
T0
�
8gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_2/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_2/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_2/while/LSTM_2/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_2/TensorArray_1<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_2/TensorArray_1*
source	gradients
�
mgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_2/TensorArray_1
�
cgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_2/TensorArrayUnstack/rangemgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0
�
`gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_2/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_2/transpose_grad/InvertPermutationInvertPermutationrnn_2/concat*
T0
�
(gradients/rnn_2/transpose_grad/transpose	Transposehgradients/rnn_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_2/transpose_grad/InvertPermutation*
Tperm0*
T0
`
2gradients/rnn_1/transpose_1_grad/InvertPermutationInvertPermutationrnn_1/concat_2*
T0
�
*gradients/rnn_1/transpose_1_grad/transpose	Transpose(gradients/rnn_2/transpose_grad/transpose2gradients/rnn_1/transpose_1_grad/InvertPermutation*
T0*
Tperm0
�
[gradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_1/TensorArrayrnn_1/while/Exit_2*$
_class
loc:@rnn_1/TensorArray*
source	gradients
�
Wgradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_1/while/Exit_2\^gradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*$
_class
loc:@rnn_1/TensorArray
�
agradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3[gradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_1/TensorArrayStack/range*gradients/rnn_1/transpose_1_grad/transposeWgradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
A
gradients/zeros_like_10	ZerosLikernn_1/while/Exit_3*
T0
A
gradients/zeros_like_11	ZerosLikernn_1/while/Exit_4*
T0
�
(gradients/rnn_1/while/Exit_2_grad/b_exitEnteragradients/rnn_1/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
(gradients/rnn_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_10*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
(gradients/rnn_1/while/Exit_4_grad/b_exitEntergradients/zeros_like_11*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
,gradients/rnn_1/while/Switch_2_grad/b_switchMerge(gradients/rnn_1/while/Exit_2_grad/b_exit3gradients/rnn_1/while/Switch_2_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_1/while/Switch_3_grad/b_switchMerge(gradients/rnn_1/while/Exit_3_grad/b_exit3gradients/rnn_1/while/Switch_3_grad_1/NextIteration*
T0*
N
�
,gradients/rnn_1/while/Switch_4_grad/b_switchMerge(gradients/rnn_1/while/Exit_4_grad/b_exit3gradients/rnn_1/while/Switch_4_grad_1/NextIteration*
T0*
N
�
)gradients/rnn_1/while/Merge_2_grad/SwitchSwitch,gradients/rnn_1/while/Switch_2_grad/b_switchgradients/b_count_22*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_2_grad/b_switch
g
3gradients/rnn_1/while/Merge_2_grad/tuple/group_depsNoOp*^gradients/rnn_1/while/Merge_2_grad/Switch
�
;gradients/rnn_1/while/Merge_2_grad/tuple/control_dependencyIdentity)gradients/rnn_1/while/Merge_2_grad/Switch4^gradients/rnn_1/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_2_grad/b_switch
�
=gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency_1Identity+gradients/rnn_1/while/Merge_2_grad/Switch:14^gradients/rnn_1/while/Merge_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_2_grad/b_switch
�
)gradients/rnn_1/while/Merge_3_grad/SwitchSwitch,gradients/rnn_1/while/Switch_3_grad/b_switchgradients/b_count_22*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_3_grad/b_switch
g
3gradients/rnn_1/while/Merge_3_grad/tuple/group_depsNoOp*^gradients/rnn_1/while/Merge_3_grad/Switch
�
;gradients/rnn_1/while/Merge_3_grad/tuple/control_dependencyIdentity)gradients/rnn_1/while/Merge_3_grad/Switch4^gradients/rnn_1/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_3_grad/b_switch
�
=gradients/rnn_1/while/Merge_3_grad/tuple/control_dependency_1Identity+gradients/rnn_1/while/Merge_3_grad/Switch:14^gradients/rnn_1/while/Merge_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_3_grad/b_switch
�
)gradients/rnn_1/while/Merge_4_grad/SwitchSwitch,gradients/rnn_1/while/Switch_4_grad/b_switchgradients/b_count_22*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_4_grad/b_switch
g
3gradients/rnn_1/while/Merge_4_grad/tuple/group_depsNoOp*^gradients/rnn_1/while/Merge_4_grad/Switch
�
;gradients/rnn_1/while/Merge_4_grad/tuple/control_dependencyIdentity)gradients/rnn_1/while/Merge_4_grad/Switch4^gradients/rnn_1/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_4_grad/b_switch
�
=gradients/rnn_1/while/Merge_4_grad/tuple/control_dependency_1Identity+gradients/rnn_1/while/Merge_4_grad/Switch:14^gradients/rnn_1/while/Merge_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_4_grad/b_switch
u
'gradients/rnn_1/while/Enter_2_grad/ExitExit;gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency*
T0
u
'gradients/rnn_1/while/Enter_3_grad/ExitExit;gradients/rnn_1/while/Merge_3_grad/tuple/control_dependency*
T0
u
'gradients/rnn_1/while/Enter_4_grad/ExitExit;gradients/rnn_1/while/Merge_4_grad/tuple/control_dependency*
T0
�
`gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3fgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter=gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency_1*+
_class!
loc:@rnn_1/while/LSTM_1/mul_2*
source	gradients
�
fgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_1/TensorArray*
T0*+
_class!
loc:@rnn_1/while/LSTM_1/mul_2*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
\gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency_1a^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@rnn_1/while/LSTM_1/mul_2
�
Pgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3`gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2\gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Vgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*)
_class
loc:@rnn_1/while/Identity_1*
valueB :
���������*
dtype0
�
Vgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Vgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn_1/while/Identity_1
�
Vgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterVgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
\gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Vgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_1/while/Identity_1^gradients/Add_5*
T0*
swap_memory( 
�
[gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2agradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
agradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterVgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�

Wgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger=^gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPopV2I^gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2;^gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2=^gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2_1I^gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV29^gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2I^gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2K^gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_17^gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV29^gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2G^gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2I^gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_15^gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPopV27^gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2\^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Ogradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp>^gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency_1Q^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityPgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3P^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ygradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity=gradients/rnn_1/while/Merge_2_grad/tuple/control_dependency_1P^gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_2_grad/b_switch
�
gradients/AddN_10AddN=gradients/rnn_1/while/Merge_4_grad/tuple/control_dependency_1Wgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_4_grad/b_switch*
N
m
-gradients/rnn_1/while/LSTM_1/mul_2_grad/ShapeShapernn_1/while/LSTM_1/Sigmoid_2*
T0*
out_type0
l
/gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape_1Shapernn_1/while/LSTM_1/Tanh_1*
T0*
out_type0
�
=gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Igradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Enter-gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape^gradients/Add_5*
swap_memory( *
T0
�
Hgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Ngradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Egradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape_1*

stack_name 
�
Egradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Kgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_1/while/LSTM_1/mul_2_grad/Shape_1^gradients/Add_5*
T0*
swap_memory( 
�
Jgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_5*
	elem_type0
�
Pgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
+gradients/rnn_1/while/LSTM_1/mul_2_grad/MulMulgradients/AddN_106gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/ConstConst*,
_class"
 loc:@rnn_1/while/LSTM_1/Tanh_1*
valueB :
���������*
dtype0
�
1gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/f_accStackV21gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/Const*
	elem_type0*,
_class"
 loc:@rnn_1/while/LSTM_1/Tanh_1*

stack_name 
�
1gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/f_acc*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
�
7gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPushV2StackPushV21gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/Enterrnn_1/while/LSTM_1/Tanh_1^gradients/Add_5*
T0*
swap_memory( 
�
6gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
<gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant(
�
+gradients/rnn_1/while/LSTM_1/mul_2_grad/SumSum+gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul=gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_1/while/LSTM_1/mul_2_grad/ReshapeReshape+gradients/rnn_1/while/LSTM_1/mul_2_grad/SumHgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1Mul8gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2gradients/AddN_10*
T0
�
3gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/ConstConst*
dtype0*/
_class%
#!loc:@rnn_1/while/LSTM_1/Sigmoid_2*
valueB :
���������
�
3gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/f_accStackV23gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/Const*/
_class%
#!loc:@rnn_1/while/LSTM_1/Sigmoid_2*

stack_name *
	elem_type0
�
3gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/EnterEnter3gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
9gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/Enterrnn_1/while/LSTM_1/Sigmoid_2^gradients/Add_5*
swap_memory( *
T0
�
8gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
>gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
-gradients/rnn_1/while/LSTM_1/mul_2_grad/Sum_1Sum-gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1?gradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape_1Reshape-gradients/rnn_1/while/LSTM_1/mul_2_grad/Sum_1Jgradients/rnn_1/while/LSTM_1/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/group_depsNoOp0^gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape2^gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape_1
�
@gradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/control_dependencyIdentity/gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape9^gradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape
�
Bgradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/control_dependency_1Identity1gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape_19^gradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_1/while/LSTM_1/mul_2_grad/Reshape_1
�
7gradients/rnn_1/while/LSTM_1/Sigmoid_2_grad/SigmoidGradSigmoidGrad8gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul_1/StackPopV2@gradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/control_dependency*
T0
�
1gradients/rnn_1/while/LSTM_1/Tanh_1_grad/TanhGradTanhGrad6gradients/rnn_1/while/LSTM_1/mul_2_grad/Mul/StackPopV2Bgradients/rnn_1/while/LSTM_1/mul_2_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_1/while/Switch_2_grad_1/NextIterationNextIterationYgradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_11AddN=gradients/rnn_1/while/Merge_3_grad/tuple/control_dependency_11gradients/rnn_1/while/LSTM_1/Tanh_1_grad/TanhGrad*
T0*?
_class5
31loc:@gradients/rnn_1/while/Switch_3_grad/b_switch*
N
g
-gradients/rnn_1/while/LSTM_1/add_1_grad/ShapeShapernn_1/while/LSTM_1/mul*
T0*
out_type0
k
/gradients/rnn_1/while/LSTM_1/add_1_grad/Shape_1Shapernn_1/while/LSTM_1/mul_1*
T0*
out_type0
�
=gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/ConstConst*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Shape*
valueB :
���������*
dtype0
�
Cgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Igradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_1/while/LSTM_1/add_1_grad/Shape^gradients/Add_5*
T0*
swap_memory( 
�
Hgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Ngradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Egradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Const_1Const*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Egradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Shape_1*

stack_name 
�
Egradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Kgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_1/while/LSTM_1/add_1_grad/Shape_1^gradients/Add_5*
swap_memory( *
T0
�
Jgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_5*
	elem_type0
�
Pgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
+gradients/rnn_1/while/LSTM_1/add_1_grad/SumSumgradients/AddN_11=gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_1/while/LSTM_1/add_1_grad/ReshapeReshape+gradients/rnn_1/while/LSTM_1/add_1_grad/SumHgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_1/while/LSTM_1/add_1_grad/Sum_1Sumgradients/AddN_11?gradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
1gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape_1Reshape-gradients/rnn_1/while/LSTM_1/add_1_grad/Sum_1Jgradients/rnn_1/while/LSTM_1/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/group_depsNoOp0^gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape2^gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape_1
�
@gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependencyIdentity/gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape9^gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape
�
Bgradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependency_1Identity1gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape_19^gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_1/while/LSTM_1/add_1_grad/Reshape_1
i
+gradients/rnn_1/while/LSTM_1/mul_grad/ShapeShapernn_1/while/LSTM_1/Sigmoid*
T0*
out_type0
g
-gradients/rnn_1/while/LSTM_1/mul_grad/Shape_1Shapernn_1/while/Identity_3*
T0*
out_type0
�
;gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2Hgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Agradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*>
_class4
20loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Shape
�
Agradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Ggradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Enter+gradients/rnn_1/while/LSTM_1/mul_grad/Shape^gradients/Add_5*
T0*
swap_memory( 
�
Fgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Lgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Cgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Const_1Const*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
Cgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Const_1*

stack_name *
	elem_type0*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Shape_1
�
Cgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Enter_1EnterCgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Igradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/Enter_1-gradients/rnn_1/while/LSTM_1/mul_grad/Shape_1^gradients/Add_5*
T0*
swap_memory( 
�
Hgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Ngradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_5*
	elem_type0
�
Ngradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterCgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
)gradients/rnn_1/while/LSTM_1/mul_grad/MulMul@gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependency4gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPopV2*
T0
�
/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/ConstConst*
dtype0*)
_class
loc:@rnn_1/while/Identity_3*
valueB :
���������
�
/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/f_accStackV2/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/Const*)
_class
loc:@rnn_1/while/Identity_3*

stack_name *
	elem_type0
�
/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/EnterEnter/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
5gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPushV2StackPushV2/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/Enterrnn_1/while/Identity_3^gradients/Add_5*
T0*
swap_memory( 
�
4gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPopV2
StackPopV2:gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
:gradients/rnn_1/while/LSTM_1/mul_grad/Mul/StackPopV2/EnterEnter/gradients/rnn_1/while/LSTM_1/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
)gradients/rnn_1/while/LSTM_1/mul_grad/SumSum)gradients/rnn_1/while/LSTM_1/mul_grad/Mul;gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn_1/while/LSTM_1/mul_grad/ReshapeReshape)gradients/rnn_1/while/LSTM_1/mul_grad/SumFgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1Mul6gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2@gradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependency*
T0
�
1gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_1/while/LSTM_1/Sigmoid*
valueB :
���������*
dtype0
�
1gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/f_accStackV21gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/Const*

stack_name *
	elem_type0*-
_class#
!loc:@rnn_1/while/LSTM_1/Sigmoid
�
1gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
7gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPushV2StackPushV21gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/Enterrnn_1/while/LSTM_1/Sigmoid^gradients/Add_5*
T0*
swap_memory( 
�
6gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2
StackPopV2<gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
<gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
+gradients/rnn_1/while/LSTM_1/mul_grad/Sum_1Sum+gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1=gradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_1/while/LSTM_1/mul_grad/Reshape_1Reshape+gradients/rnn_1/while/LSTM_1/mul_grad/Sum_1Hgradients/rnn_1/while/LSTM_1/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
6gradients/rnn_1/while/LSTM_1/mul_grad/tuple/group_depsNoOp.^gradients/rnn_1/while/LSTM_1/mul_grad/Reshape0^gradients/rnn_1/while/LSTM_1/mul_grad/Reshape_1
�
>gradients/rnn_1/while/LSTM_1/mul_grad/tuple/control_dependencyIdentity-gradients/rnn_1/while/LSTM_1/mul_grad/Reshape7^gradients/rnn_1/while/LSTM_1/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Reshape
�
@gradients/rnn_1/while/LSTM_1/mul_grad/tuple/control_dependency_1Identity/gradients/rnn_1/while/LSTM_1/mul_grad/Reshape_17^gradients/rnn_1/while/LSTM_1/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_grad/Reshape_1
m
-gradients/rnn_1/while/LSTM_1/mul_1_grad/ShapeShapernn_1/while/LSTM_1/Sigmoid_1*
T0*
out_type0
j
/gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape_1Shapernn_1/while/LSTM_1/Tanh*
T0*
out_type0
�
=gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2Jgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Cgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape*
valueB :
���������
�
Cgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_accStackV2Cgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Const*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape*

stack_name *
	elem_type0
�
Cgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/EnterEnterCgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Igradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Cgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Enter-gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape^gradients/Add_5*
T0*
swap_memory( 
�
Hgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Ngradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Ngradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterCgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Egradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape_1*
valueB :
���������
�
Egradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Egradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Const_1*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape_1*

stack_name *
	elem_type0
�
Egradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Enter_1EnterEgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
�
Kgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Egradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/Enter_1/gradients/rnn_1/while/LSTM_1/mul_1_grad/Shape_1^gradients/Add_5*
T0*
swap_memory( 
�
Jgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Pgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_5*
	elem_type0
�
Pgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterEgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
+gradients/rnn_1/while/LSTM_1/mul_1_grad/MulMulBgradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependency_16gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV2*
T0
�
1gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/ConstConst**
_class 
loc:@rnn_1/while/LSTM_1/Tanh*
valueB :
���������*
dtype0
�
1gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/f_accStackV21gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/Const*

stack_name *
	elem_type0**
_class 
loc:@rnn_1/while/LSTM_1/Tanh
�
1gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/f_acc*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
�
7gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPushV2StackPushV21gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/Enterrnn_1/while/LSTM_1/Tanh^gradients/Add_5*
T0*
swap_memory( 
�
6gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV2
StackPopV2<gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
<gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV2/EnterEnter1gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant(
�
+gradients/rnn_1/while/LSTM_1/mul_1_grad/SumSum+gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul=gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/rnn_1/while/LSTM_1/mul_1_grad/ReshapeReshape+gradients/rnn_1/while/LSTM_1/mul_1_grad/SumHgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
-gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1Mul8gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2Bgradients/rnn_1/while/LSTM_1/add_1_grad/tuple/control_dependency_1*
T0
�
3gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/ConstConst*
dtype0*/
_class%
#!loc:@rnn_1/while/LSTM_1/Sigmoid_1*
valueB :
���������
�
3gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/f_accStackV23gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/Const*/
_class%
#!loc:@rnn_1/while/LSTM_1/Sigmoid_1*

stack_name *
	elem_type0
�
3gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/EnterEnter3gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
9gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPushV2StackPushV23gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/Enterrnn_1/while/LSTM_1/Sigmoid_1^gradients/Add_5*
swap_memory( *
T0
�
8gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2
StackPopV2>gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
>gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2/EnterEnter3gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
-gradients/rnn_1/while/LSTM_1/mul_1_grad/Sum_1Sum-gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1?gradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape_1Reshape-gradients/rnn_1/while/LSTM_1/mul_1_grad/Sum_1Jgradients/rnn_1/while/LSTM_1/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
8gradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/group_depsNoOp0^gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape2^gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape_1
�
@gradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/control_dependencyIdentity/gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape9^gradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape
�
Bgradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/control_dependency_1Identity1gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape_19^gradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_1/while/LSTM_1/mul_1_grad/Reshape_1
�
5gradients/rnn_1/while/LSTM_1/Sigmoid_grad/SigmoidGradSigmoidGrad6gradients/rnn_1/while/LSTM_1/mul_grad/Mul_1/StackPopV2>gradients/rnn_1/while/LSTM_1/mul_grad/tuple/control_dependency*
T0
�
7gradients/rnn_1/while/LSTM_1/Sigmoid_1_grad/SigmoidGradSigmoidGrad8gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul_1/StackPopV2@gradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/control_dependency*
T0
�
/gradients/rnn_1/while/LSTM_1/Tanh_grad/TanhGradTanhGrad6gradients/rnn_1/while/LSTM_1/mul_1_grad/Mul/StackPopV2Bgradients/rnn_1/while/LSTM_1/mul_1_grad/tuple/control_dependency_1*
T0
i
+gradients/rnn_1/while/LSTM_1/add_grad/ShapeShapernn_1/while/LSTM_1/split:2*
T0*
out_type0
h
-gradients/rnn_1/while/LSTM_1/add_grad/Shape_1Const^gradients/Sub_5*
valueB *
dtype0
�
;gradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2-gradients/rnn_1/while/LSTM_1/add_grad/Shape_1*
T0
�
Agradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/ConstConst*>
_class4
20loc:@gradients/rnn_1/while/LSTM_1/add_grad/Shape*
valueB :
���������*
dtype0
�
Agradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/f_accStackV2Agradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/rnn_1/while/LSTM_1/add_grad/Shape*

stack_name *
	elem_type0
�
Agradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/EnterEnterAgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
Ggradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/Enter+gradients/rnn_1/while/LSTM_1/add_grad/Shape^gradients/Add_5*
swap_memory( *
T0
�
Fgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Lgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Lgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterAgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant(
�
)gradients/rnn_1/while/LSTM_1/add_grad/SumSum5gradients/rnn_1/while/LSTM_1/Sigmoid_grad/SigmoidGrad;gradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn_1/while/LSTM_1/add_grad/ReshapeReshape)gradients/rnn_1/while/LSTM_1/add_grad/SumFgradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
+gradients/rnn_1/while/LSTM_1/add_grad/Sum_1Sum5gradients/rnn_1/while/LSTM_1/Sigmoid_grad/SigmoidGrad=gradients/rnn_1/while/LSTM_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
/gradients/rnn_1/while/LSTM_1/add_grad/Reshape_1Reshape+gradients/rnn_1/while/LSTM_1/add_grad/Sum_1-gradients/rnn_1/while/LSTM_1/add_grad/Shape_1*
T0*
Tshape0
�
6gradients/rnn_1/while/LSTM_1/add_grad/tuple/group_depsNoOp.^gradients/rnn_1/while/LSTM_1/add_grad/Reshape0^gradients/rnn_1/while/LSTM_1/add_grad/Reshape_1
�
>gradients/rnn_1/while/LSTM_1/add_grad/tuple/control_dependencyIdentity-gradients/rnn_1/while/LSTM_1/add_grad/Reshape7^gradients/rnn_1/while/LSTM_1/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn_1/while/LSTM_1/add_grad/Reshape
�
@gradients/rnn_1/while/LSTM_1/add_grad/tuple/control_dependency_1Identity/gradients/rnn_1/while/LSTM_1/add_grad/Reshape_17^gradients/rnn_1/while/LSTM_1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/add_grad/Reshape_1
�
3gradients/rnn_1/while/Switch_3_grad_1/NextIterationNextIteration@gradients/rnn_1/while/LSTM_1/mul_grad/tuple/control_dependency_1*
T0
�
.gradients/rnn_1/while/LSTM_1/split_grad/concatConcatV27gradients/rnn_1/while/LSTM_1/Sigmoid_1_grad/SigmoidGrad/gradients/rnn_1/while/LSTM_1/Tanh_grad/TanhGrad>gradients/rnn_1/while/LSTM_1/add_grad/tuple/control_dependency7gradients/rnn_1/while/LSTM_1/Sigmoid_2_grad/SigmoidGrad4gradients/rnn_1/while/LSTM_1/split_grad/concat/Const*
N*

Tidx0*
T0
p
4gradients/rnn_1/while/LSTM_1/split_grad/concat/ConstConst^gradients/Sub_5*
dtype0*
value	B :
�
5gradients/rnn_1/while/LSTM_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/rnn_1/while/LSTM_1/split_grad/concat*
T0*
data_formatNHWC
�
:gradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/group_depsNoOp6^gradients/rnn_1/while/LSTM_1/BiasAdd_grad/BiasAddGrad/^gradients/rnn_1/while/LSTM_1/split_grad/concat
�
Bgradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/rnn_1/while/LSTM_1/split_grad/concat;^gradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_1/while/LSTM_1/split_grad/concat
�
Dgradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/rnn_1/while/LSTM_1/BiasAdd_grad/BiasAddGrad;^gradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn_1/while/LSTM_1/BiasAdd_grad/BiasAddGrad
�
/gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMulMatMulBgradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/control_dependency5gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
T0
�
5gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul/EnterEnterrnn/LSTM_1/kernel/read*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant(
�
1gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1MatMul<gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPopV2Bgradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
7gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/ConstConst*,
_class"
 loc:@rnn_1/while/LSTM_1/concat*
valueB :
���������*
dtype0
�
7gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/f_accStackV27gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/Const*
	elem_type0*,
_class"
 loc:@rnn_1/while/LSTM_1/concat*

stack_name 
�
7gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/EnterEnter7gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
=gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPushV2StackPushV27gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/Enterrnn_1/while/LSTM_1/concat^gradients/Add_5*
swap_memory( *
T0
�
<gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Bgradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
Bgradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/StackPopV2/EnterEnter7gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
9gradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/group_depsNoOp0^gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul2^gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1
�
Agradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/control_dependencyIdentity/gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul:^gradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul
�
Cgradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/control_dependency_1Identity1gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1:^gradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn_1/while/LSTM_1/MatMul_grad/MatMul_1
g
5gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_1Enter5gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_2Merge7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_1=gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
6gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/SwitchSwitch7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_2gradients/b_count_22*
T0
�
3gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/AddAdd8gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/Switch:1Dgradients/rnn_1/while/LSTM_1/BiasAdd_grad/tuple/control_dependency_1*
T0
�
=gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/NextIterationNextIteration3gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/Add*
T0
�
7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_3Exit6gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/Switch*
T0
j
.gradients/rnn_1/while/LSTM_1/concat_grad/ConstConst^gradients/Sub_5*
value	B :*
dtype0
i
-gradients/rnn_1/while/LSTM_1/concat_grad/RankConst^gradients/Sub_5*
value	B :*
dtype0
�
,gradients/rnn_1/while/LSTM_1/concat_grad/modFloorMod.gradients/rnn_1/while/LSTM_1/concat_grad/Const-gradients/rnn_1/while/LSTM_1/concat_grad/Rank*
T0
o
.gradients/rnn_1/while/LSTM_1/concat_grad/ShapeShapernn_1/while/TensorArrayReadV3*
T0*
out_type0
�
/gradients/rnn_1/while/LSTM_1/concat_grad/ShapeNShapeN:gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2<gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N
�
5gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/ConstConst*0
_class&
$"loc:@rnn_1/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
5gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_accStackV25gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Const*0
_class&
$"loc:@rnn_1/while/TensorArrayReadV3*

stack_name *
	elem_type0
�
5gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/EnterEnter5gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *)

frame_namernn_1/while/while_context
�
;gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPushV2StackPushV25gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Enterrnn_1/while/TensorArrayReadV3^gradients/Add_5*
T0*
swap_memory( 
�
:gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2
StackPopV2@gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_5*
	elem_type0
�
@gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2/EnterEnter5gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
7gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Const_1Const*
dtype0*)
_class
loc:@rnn_1/while/Identity_4*
valueB :
���������
�
7gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_acc_1StackV27gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Const_1*)
_class
loc:@rnn_1/while/Identity_4*

stack_name *
	elem_type0
�
7gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Enter_1Enter7gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_acc_1*
parallel_iterations *)

frame_namernn_1/while/while_context*
T0*
is_constant(
�
=gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPushV2_1StackPushV27gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/Enter_1rnn_1/while/Identity_4^gradients/Add_5*
T0*
swap_memory( 
�
<gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2_1
StackPopV2Bgradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_5*
	elem_type0
�
Bgradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/StackPopV2_1/EnterEnter7gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
5gradients/rnn_1/while/LSTM_1/concat_grad/ConcatOffsetConcatOffset,gradients/rnn_1/while/LSTM_1/concat_grad/mod/gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN1gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN:1*
N
�
.gradients/rnn_1/while/LSTM_1/concat_grad/SliceSliceAgradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/control_dependency5gradients/rnn_1/while/LSTM_1/concat_grad/ConcatOffset/gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN*
T0*
Index0
�
0gradients/rnn_1/while/LSTM_1/concat_grad/Slice_1SliceAgradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/control_dependency7gradients/rnn_1/while/LSTM_1/concat_grad/ConcatOffset:11gradients/rnn_1/while/LSTM_1/concat_grad/ShapeN:1*
T0*
Index0
�
9gradients/rnn_1/while/LSTM_1/concat_grad/tuple/group_depsNoOp/^gradients/rnn_1/while/LSTM_1/concat_grad/Slice1^gradients/rnn_1/while/LSTM_1/concat_grad/Slice_1
�
Agradients/rnn_1/while/LSTM_1/concat_grad/tuple/control_dependencyIdentity.gradients/rnn_1/while/LSTM_1/concat_grad/Slice:^gradients/rnn_1/while/LSTM_1/concat_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn_1/while/LSTM_1/concat_grad/Slice
�
Cgradients/rnn_1/while/LSTM_1/concat_grad/tuple/control_dependency_1Identity0gradients/rnn_1/while/LSTM_1/concat_grad/Slice_1:^gradients/rnn_1/while/LSTM_1/concat_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn_1/while/LSTM_1/concat_grad/Slice_1
k
4gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_1Enter4gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant( 
�
6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_2Merge6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_1<gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/NextIteration*
T0*
N
�
5gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/SwitchSwitch6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_2gradients/b_count_22*
T0
�
2gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/AddAdd7gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/Switch:1Cgradients/rnn_1/while/LSTM_1/MatMul_grad/tuple/control_dependency_1*
T0
�
<gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/NextIterationNextIteration2gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/Add*
T0
~
6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_3Exit5gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/Switch*
T0
�
Ngradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Tgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterVgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_5*6
_class,
*(loc:@rnn_1/while/TensorArrayReadV3/Enter*
source	gradients
�
Tgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_1/TensorArray_1*
T0*6
_class,
*(loc:@rnn_1/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Vgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter@rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6
_class,
*(loc:@rnn_1/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context
�
Jgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityVgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1O^gradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn_1/while/TensorArrayReadV3/Enter
�
Pgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ngradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3[gradients/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Agradients/rnn_1/while/LSTM_1/concat_grad/tuple/control_dependencyJgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
g
:gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter:gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *3

frame_name%#gradients/rnn_1/while/while_context*
T0*
is_constant( 
�
<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Bgradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
;gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_22*
T0
�
8gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/AddAdd=gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1Pgradients/rnn_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Bgradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration8gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit;gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
3gradients/rnn_1/while/Switch_4_grad_1/NextIterationNextIterationCgradients/rnn_1/while/LSTM_1/concat_grad/tuple/control_dependency_1*
T0
�
qgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_1/TensorArray_1<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*&
_class
loc:@rnn_1/TensorArray_1*
source	gradients
�
mgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3r^gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn_1/TensorArray_1
�
cgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3qgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3rnn_1/TensorArrayUnstack/rangemgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
element_shape:
�
`gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpd^gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3=^gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
hgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitycgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3a^gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
jgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity<gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3a^gradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/rnn_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
\
0gradients/rnn_1/transpose_grad/InvertPermutationInvertPermutationrnn_1/concat*
T0
�
(gradients/rnn_1/transpose_grad/transpose	Transposehgradients/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency0gradients/rnn_1/transpose_grad/InvertPermutation*
Tperm0*
T0
\
0gradients/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn/concat_2*
T0
�
(gradients/rnn/transpose_1_grad/transpose	Transpose(gradients/rnn_1/transpose_grad/transpose0gradients/rnn/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_2*"
_class
loc:@rnn/TensorArray*
source	gradients
�
Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_2Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*"
_class
loc:@rnn/TensorArray
�
_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range(gradients/rnn/transpose_1_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
?
gradients/zeros_like_12	ZerosLikernn/while/Exit_3*
T0
?
gradients/zeros_like_13	ZerosLikernn/while/Exit_4*
T0
�
&gradients/rnn/while/Exit_2_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_12*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_like_13*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N
�
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N
�
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration*
N*
T0
�
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_26*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
c
1gradients/rnn/while/Merge_2_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_2_grad/Switch
�
9gradients/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_2_grad/Switch2^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
�
;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_2_grad/Switch:12^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
�
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_26*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
c
1gradients/rnn/while/Merge_3_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_3_grad/Switch
�
9gradients/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_3_grad/Switch2^gradients/rnn/while/Merge_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
�
;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_3_grad/Switch:12^gradients/rnn/while/Merge_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
�
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_26*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
c
1gradients/rnn/while/Merge_4_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_4_grad/Switch
�
9gradients/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_4_grad/Switch2^gradients/rnn/while/Merge_4_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
�
;gradients/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_4_grad/Switch:12^gradients/rnn/while/Merge_4_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
q
%gradients/rnn/while/Enter_2_grad/ExitExit9gradients/rnn/while/Merge_2_grad/tuple/control_dependency*
T0
q
%gradients/rnn/while/Enter_3_grad/ExitExit9gradients/rnn/while/Merge_3_grad/tuple/control_dependency*
T0
q
%gradients/rnn/while/Enter_4_grad/ExitExit9gradients/rnn/while/Merge_4_grad/tuple/control_dependency*
T0
�
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1*'
_class
loc:@rnn/while/LSTM/mul_2*
source	gradients
�
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
T0*'
_class
loc:@rnn/while/LSTM/mul_2*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@rnn/while/LSTM/mul_2
�
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
dtype0*'
_class
loc:@rnn/while/Identity_1*
valueB :
���������
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
	elem_type0*'
_class
loc:@rnn/while/Identity_1
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity_1^gradients/Add_6*
T0*
swap_memory( 
�
Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�

Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger9^gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPopV2E^gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2G^gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1C^gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV27^gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV29^gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2_1E^gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2G^gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_13^gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV25^gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2E^gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2G^gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_13^gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV25^gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2C^gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2E^gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_11^gradients/rnn/while/LSTM/mul_grad/Mul/StackPopV23^gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2Z^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
�
Mgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp<^gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1O^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Wgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
�
gradients/AddN_12AddN;gradients/rnn/while/Merge_4_grad/tuple/control_dependency_1Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
N
e
)gradients/rnn/while/LSTM/mul_2_grad/ShapeShapernn/while/LSTM/Sigmoid_2*
T0*
out_type0
d
+gradients/rnn/while/LSTM/mul_2_grad/Shape_1Shapernn/while/LSTM/Tanh_1*
T0*
out_type0
�
9gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2Fgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/ConstConst*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_2_grad/Shape*
valueB :
���������*
dtype0
�
?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_accStackV2?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Const*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_2_grad/Shape*

stack_name *
	elem_type0
�
?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/EnterEnter?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Egradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Enter)gradients/rnn/while/LSTM/mul_2_grad/Shape^gradients/Add_6*
swap_memory( *
T0
�
Dgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Jgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
Jgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter?gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
Agradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Const_1Const*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Agradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Agradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Const_1*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_2_grad/Shape_1*

stack_name *
	elem_type0
�
Agradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Enter_1EnterAgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Ggradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Agradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/Enter_1+gradients/rnn/while/LSTM/mul_2_grad/Shape_1^gradients/Add_6*
swap_memory( *
T0
�
Fgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Lgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_6*
	elem_type0
�
Lgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterAgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
~
'gradients/rnn/while/LSTM/mul_2_grad/MulMulgradients/AddN_122gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV2*
T0
�
-gradients/rnn/while/LSTM/mul_2_grad/Mul/ConstConst*(
_class
loc:@rnn/while/LSTM/Tanh_1*
valueB :
���������*
dtype0
�
-gradients/rnn/while/LSTM/mul_2_grad/Mul/f_accStackV2-gradients/rnn/while/LSTM/mul_2_grad/Mul/Const*(
_class
loc:@rnn/while/LSTM/Tanh_1*

stack_name *
	elem_type0
�
-gradients/rnn/while/LSTM/mul_2_grad/Mul/EnterEnter-gradients/rnn/while/LSTM/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
3gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPushV2StackPushV2-gradients/rnn/while/LSTM/mul_2_grad/Mul/Enterrnn/while/LSTM/Tanh_1^gradients/Add_6*
T0*
swap_memory( 
�
2gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV2
StackPopV28gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
8gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV2/EnterEnter-gradients/rnn/while/LSTM/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
'gradients/rnn/while/LSTM/mul_2_grad/SumSum'gradients/rnn/while/LSTM/mul_2_grad/Mul9gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
+gradients/rnn/while/LSTM/mul_2_grad/ReshapeReshape'gradients/rnn/while/LSTM/mul_2_grad/SumDgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
)gradients/rnn/while/LSTM/mul_2_grad/Mul_1Mul4gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2gradients/AddN_12*
T0
�
/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/ConstConst*
dtype0*+
_class!
loc:@rnn/while/LSTM/Sigmoid_2*
valueB :
���������
�
/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/f_accStackV2/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/Const*+
_class!
loc:@rnn/while/LSTM/Sigmoid_2*

stack_name *
	elem_type0
�
/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/EnterEnter/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/f_acc*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
5gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPushV2StackPushV2/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/Enterrnn/while/LSTM/Sigmoid_2^gradients/Add_6*
swap_memory( *
T0
�
4gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2
StackPopV2:gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
:gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2/EnterEnter/gradients/rnn/while/LSTM/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
)gradients/rnn/while/LSTM/mul_2_grad/Sum_1Sum)gradients/rnn/while/LSTM/mul_2_grad/Mul_1;gradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn/while/LSTM/mul_2_grad/Reshape_1Reshape)gradients/rnn/while/LSTM/mul_2_grad/Sum_1Fgradients/rnn/while/LSTM/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
4gradients/rnn/while/LSTM/mul_2_grad/tuple/group_depsNoOp,^gradients/rnn/while/LSTM/mul_2_grad/Reshape.^gradients/rnn/while/LSTM/mul_2_grad/Reshape_1
�
<gradients/rnn/while/LSTM/mul_2_grad/tuple/control_dependencyIdentity+gradients/rnn/while/LSTM/mul_2_grad/Reshape5^gradients/rnn/while/LSTM/mul_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_2_grad/Reshape
�
>gradients/rnn/while/LSTM/mul_2_grad/tuple/control_dependency_1Identity-gradients/rnn/while/LSTM/mul_2_grad/Reshape_15^gradients/rnn/while/LSTM/mul_2_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/LSTM/mul_2_grad/Reshape_1
�
3gradients/rnn/while/LSTM/Sigmoid_2_grad/SigmoidGradSigmoidGrad4gradients/rnn/while/LSTM/mul_2_grad/Mul_1/StackPopV2<gradients/rnn/while/LSTM/mul_2_grad/tuple/control_dependency*
T0
�
-gradients/rnn/while/LSTM/Tanh_1_grad/TanhGradTanhGrad2gradients/rnn/while/LSTM/mul_2_grad/Mul/StackPopV2>gradients/rnn/while/LSTM/mul_2_grad/tuple/control_dependency_1*
T0
�
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationWgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_13AddN;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1-gradients/rnn/while/LSTM/Tanh_1_grad/TanhGrad*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
N
_
)gradients/rnn/while/LSTM/add_1_grad/ShapeShapernn/while/LSTM/mul*
T0*
out_type0
c
+gradients/rnn/while/LSTM/add_1_grad/Shape_1Shapernn/while/LSTM/mul_1*
T0*
out_type0
�
9gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2Fgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*<
_class2
0.loc:@gradients/rnn/while/LSTM/add_1_grad/Shape*
valueB :
���������
�
?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_accStackV2?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*<
_class2
0.loc:@gradients/rnn/while/LSTM/add_1_grad/Shape*

stack_name 
�
?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/EnterEnter?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Egradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Enter)gradients/rnn/while/LSTM/add_1_grad/Shape^gradients/Add_6*
T0*
swap_memory( 
�
Dgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Jgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
Jgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter?gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
Agradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*>
_class4
20loc:@gradients/rnn/while/LSTM/add_1_grad/Shape_1*
valueB :
���������
�
Agradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Agradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*>
_class4
20loc:@gradients/rnn/while/LSTM/add_1_grad/Shape_1*

stack_name 
�
Agradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Enter_1EnterAgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Ggradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Agradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/Enter_1+gradients/rnn/while/LSTM/add_1_grad/Shape_1^gradients/Add_6*
swap_memory( *
T0
�
Fgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Lgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_6*
	elem_type0
�
Lgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterAgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
'gradients/rnn/while/LSTM/add_1_grad/SumSumgradients/AddN_139gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
+gradients/rnn/while/LSTM/add_1_grad/ReshapeReshape'gradients/rnn/while/LSTM/add_1_grad/SumDgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
)gradients/rnn/while/LSTM/add_1_grad/Sum_1Sumgradients/AddN_13;gradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
-gradients/rnn/while/LSTM/add_1_grad/Reshape_1Reshape)gradients/rnn/while/LSTM/add_1_grad/Sum_1Fgradients/rnn/while/LSTM/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
4gradients/rnn/while/LSTM/add_1_grad/tuple/group_depsNoOp,^gradients/rnn/while/LSTM/add_1_grad/Reshape.^gradients/rnn/while/LSTM/add_1_grad/Reshape_1
�
<gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependencyIdentity+gradients/rnn/while/LSTM/add_1_grad/Reshape5^gradients/rnn/while/LSTM/add_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/add_1_grad/Reshape
�
>gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependency_1Identity-gradients/rnn/while/LSTM/add_1_grad/Reshape_15^gradients/rnn/while/LSTM/add_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/LSTM/add_1_grad/Reshape_1
a
'gradients/rnn/while/LSTM/mul_grad/ShapeShapernn/while/LSTM/Sigmoid*
T0*
out_type0
a
)gradients/rnn/while/LSTM/mul_grad/Shape_1Shapernn/while/Identity_3*
T0*
out_type0
�
7gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2Dgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/ConstConst*:
_class0
.,loc:@gradients/rnn/while/LSTM/mul_grad/Shape*
valueB :
���������*
dtype0
�
=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_accStackV2=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Const*:
_class0
.,loc:@gradients/rnn/while/LSTM/mul_grad/Shape*

stack_name *
	elem_type0
�
=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/EnterEnter=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
Cgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Enter'gradients/rnn/while/LSTM/mul_grad/Shape^gradients/Add_6*
T0*
swap_memory( 
�
Bgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Hgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
Hgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter=gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
�
?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Const_1Const*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_acc_1StackV2?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_grad/Shape_1*

stack_name 
�
?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Enter_1Enter?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Egradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/Enter_1)gradients/rnn/while/LSTM/mul_grad/Shape_1^gradients/Add_6*
T0*
swap_memory( 
�
Dgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Jgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_6*
	elem_type0
�
Jgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter?gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
%gradients/rnn/while/LSTM/mul_grad/MulMul<gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependency0gradients/rnn/while/LSTM/mul_grad/Mul/StackPopV2*
T0
�
+gradients/rnn/while/LSTM/mul_grad/Mul/ConstConst*'
_class
loc:@rnn/while/Identity_3*
valueB :
���������*
dtype0
�
+gradients/rnn/while/LSTM/mul_grad/Mul/f_accStackV2+gradients/rnn/while/LSTM/mul_grad/Mul/Const*
	elem_type0*'
_class
loc:@rnn/while/Identity_3*

stack_name 
�
+gradients/rnn/while/LSTM/mul_grad/Mul/EnterEnter+gradients/rnn/while/LSTM/mul_grad/Mul/f_acc*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
1gradients/rnn/while/LSTM/mul_grad/Mul/StackPushV2StackPushV2+gradients/rnn/while/LSTM/mul_grad/Mul/Enterrnn/while/Identity_3^gradients/Add_6*
T0*
swap_memory( 
�
0gradients/rnn/while/LSTM/mul_grad/Mul/StackPopV2
StackPopV26gradients/rnn/while/LSTM/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
6gradients/rnn/while/LSTM/mul_grad/Mul/StackPopV2/EnterEnter+gradients/rnn/while/LSTM/mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
%gradients/rnn/while/LSTM/mul_grad/SumSum%gradients/rnn/while/LSTM/mul_grad/Mul7gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
)gradients/rnn/while/LSTM/mul_grad/ReshapeReshape%gradients/rnn/while/LSTM/mul_grad/SumBgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
'gradients/rnn/while/LSTM/mul_grad/Mul_1Mul2gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2<gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependency*
T0
�
-gradients/rnn/while/LSTM/mul_grad/Mul_1/ConstConst*)
_class
loc:@rnn/while/LSTM/Sigmoid*
valueB :
���������*
dtype0
�
-gradients/rnn/while/LSTM/mul_grad/Mul_1/f_accStackV2-gradients/rnn/while/LSTM/mul_grad/Mul_1/Const*

stack_name *
	elem_type0*)
_class
loc:@rnn/while/LSTM/Sigmoid
�
-gradients/rnn/while/LSTM/mul_grad/Mul_1/EnterEnter-gradients/rnn/while/LSTM/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
3gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPushV2StackPushV2-gradients/rnn/while/LSTM/mul_grad/Mul_1/Enterrnn/while/LSTM/Sigmoid^gradients/Add_6*
T0*
swap_memory( 
�
2gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2
StackPopV28gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
8gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2/EnterEnter-gradients/rnn/while/LSTM/mul_grad/Mul_1/f_acc*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
�
'gradients/rnn/while/LSTM/mul_grad/Sum_1Sum'gradients/rnn/while/LSTM/mul_grad/Mul_19gradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
+gradients/rnn/while/LSTM/mul_grad/Reshape_1Reshape'gradients/rnn/while/LSTM/mul_grad/Sum_1Dgradients/rnn/while/LSTM/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
2gradients/rnn/while/LSTM/mul_grad/tuple/group_depsNoOp*^gradients/rnn/while/LSTM/mul_grad/Reshape,^gradients/rnn/while/LSTM/mul_grad/Reshape_1
�
:gradients/rnn/while/LSTM/mul_grad/tuple/control_dependencyIdentity)gradients/rnn/while/LSTM/mul_grad/Reshape3^gradients/rnn/while/LSTM/mul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_grad/Reshape
�
<gradients/rnn/while/LSTM/mul_grad/tuple/control_dependency_1Identity+gradients/rnn/while/LSTM/mul_grad/Reshape_13^gradients/rnn/while/LSTM/mul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_grad/Reshape_1
e
)gradients/rnn/while/LSTM/mul_1_grad/ShapeShapernn/while/LSTM/Sigmoid_1*
T0*
out_type0
b
+gradients/rnn/while/LSTM/mul_1_grad/Shape_1Shapernn/while/LSTM/Tanh*
T0*
out_type0
�
9gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2Fgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_1_grad/Shape*
valueB :
���������
�
?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_accStackV2?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
	elem_type0*<
_class2
0.loc:@gradients/rnn/while/LSTM/mul_1_grad/Shape
�
?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/EnterEnter?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Egradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Enter)gradients/rnn/while/LSTM/mul_1_grad/Shape^gradients/Add_6*
swap_memory( *
T0
�
Dgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Jgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
Jgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter?gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
Agradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Const_1Const*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Agradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Agradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Const_1*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_1_grad/Shape_1*

stack_name *
	elem_type0
�
Agradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Enter_1EnterAgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Ggradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Agradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/Enter_1+gradients/rnn/while/LSTM/mul_1_grad/Shape_1^gradients/Add_6*
T0*
swap_memory( 
�
Fgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Lgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_6*
	elem_type0
�
Lgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterAgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
'gradients/rnn/while/LSTM/mul_1_grad/MulMul>gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependency_12gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV2*
T0
�
-gradients/rnn/while/LSTM/mul_1_grad/Mul/ConstConst*
dtype0*&
_class
loc:@rnn/while/LSTM/Tanh*
valueB :
���������
�
-gradients/rnn/while/LSTM/mul_1_grad/Mul/f_accStackV2-gradients/rnn/while/LSTM/mul_1_grad/Mul/Const*
	elem_type0*&
_class
loc:@rnn/while/LSTM/Tanh*

stack_name 
�
-gradients/rnn/while/LSTM/mul_1_grad/Mul/EnterEnter-gradients/rnn/while/LSTM/mul_1_grad/Mul/f_acc*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
3gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPushV2StackPushV2-gradients/rnn/while/LSTM/mul_1_grad/Mul/Enterrnn/while/LSTM/Tanh^gradients/Add_6*
T0*
swap_memory( 
�
2gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV2
StackPopV28gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
8gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV2/EnterEnter-gradients/rnn/while/LSTM/mul_1_grad/Mul/f_acc*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context*
T0*
is_constant(
�
'gradients/rnn/while/LSTM/mul_1_grad/SumSum'gradients/rnn/while/LSTM/mul_1_grad/Mul9gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
+gradients/rnn/while/LSTM/mul_1_grad/ReshapeReshape'gradients/rnn/while/LSTM/mul_1_grad/SumDgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
)gradients/rnn/while/LSTM/mul_1_grad/Mul_1Mul4gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2>gradients/rnn/while/LSTM/add_1_grad/tuple/control_dependency_1*
T0
�
/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/ConstConst*+
_class!
loc:@rnn/while/LSTM/Sigmoid_1*
valueB :
���������*
dtype0
�
/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/f_accStackV2/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/Const*
	elem_type0*+
_class!
loc:@rnn/while/LSTM/Sigmoid_1*

stack_name 
�
/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/EnterEnter/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
5gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPushV2StackPushV2/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/Enterrnn/while/LSTM/Sigmoid_1^gradients/Add_6*
T0*
swap_memory( 
�
4gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2
StackPopV2:gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
:gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2/EnterEnter/gradients/rnn/while/LSTM/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
)gradients/rnn/while/LSTM/mul_1_grad/Sum_1Sum)gradients/rnn/while/LSTM/mul_1_grad/Mul_1;gradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
-gradients/rnn/while/LSTM/mul_1_grad/Reshape_1Reshape)gradients/rnn/while/LSTM/mul_1_grad/Sum_1Fgradients/rnn/while/LSTM/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
4gradients/rnn/while/LSTM/mul_1_grad/tuple/group_depsNoOp,^gradients/rnn/while/LSTM/mul_1_grad/Reshape.^gradients/rnn/while/LSTM/mul_1_grad/Reshape_1
�
<gradients/rnn/while/LSTM/mul_1_grad/tuple/control_dependencyIdentity+gradients/rnn/while/LSTM/mul_1_grad/Reshape5^gradients/rnn/while/LSTM/mul_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/mul_1_grad/Reshape
�
>gradients/rnn/while/LSTM/mul_1_grad/tuple/control_dependency_1Identity-gradients/rnn/while/LSTM/mul_1_grad/Reshape_15^gradients/rnn/while/LSTM/mul_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/LSTM/mul_1_grad/Reshape_1
�
1gradients/rnn/while/LSTM/Sigmoid_grad/SigmoidGradSigmoidGrad2gradients/rnn/while/LSTM/mul_grad/Mul_1/StackPopV2:gradients/rnn/while/LSTM/mul_grad/tuple/control_dependency*
T0
�
3gradients/rnn/while/LSTM/Sigmoid_1_grad/SigmoidGradSigmoidGrad4gradients/rnn/while/LSTM/mul_1_grad/Mul_1/StackPopV2<gradients/rnn/while/LSTM/mul_1_grad/tuple/control_dependency*
T0
�
+gradients/rnn/while/LSTM/Tanh_grad/TanhGradTanhGrad2gradients/rnn/while/LSTM/mul_1_grad/Mul/StackPopV2>gradients/rnn/while/LSTM/mul_1_grad/tuple/control_dependency_1*
T0
a
'gradients/rnn/while/LSTM/add_grad/ShapeShapernn/while/LSTM/split:2*
T0*
out_type0
d
)gradients/rnn/while/LSTM/add_grad/Shape_1Const^gradients/Sub_6*
valueB *
dtype0
�
7gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV2)gradients/rnn/while/LSTM/add_grad/Shape_1*
T0
�
=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/ConstConst*
dtype0*:
_class0
.,loc:@gradients/rnn/while/LSTM/add_grad/Shape*
valueB :
���������
�
=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/f_accStackV2=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/Const*
	elem_type0*:
_class0
.,loc:@gradients/rnn/while/LSTM/add_grad/Shape*

stack_name 
�
=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/EnterEnter=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
Cgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/Enter'gradients/rnn/while/LSTM/add_grad/Shape^gradients/Add_6*
T0*
swap_memory( 
�
Bgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Hgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
Hgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter=gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
%gradients/rnn/while/LSTM/add_grad/SumSum1gradients/rnn/while/LSTM/Sigmoid_grad/SigmoidGrad7gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
)gradients/rnn/while/LSTM/add_grad/ReshapeReshape%gradients/rnn/while/LSTM/add_grad/SumBgradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
'gradients/rnn/while/LSTM/add_grad/Sum_1Sum1gradients/rnn/while/LSTM/Sigmoid_grad/SigmoidGrad9gradients/rnn/while/LSTM/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
+gradients/rnn/while/LSTM/add_grad/Reshape_1Reshape'gradients/rnn/while/LSTM/add_grad/Sum_1)gradients/rnn/while/LSTM/add_grad/Shape_1*
T0*
Tshape0
�
2gradients/rnn/while/LSTM/add_grad/tuple/group_depsNoOp*^gradients/rnn/while/LSTM/add_grad/Reshape,^gradients/rnn/while/LSTM/add_grad/Reshape_1
�
:gradients/rnn/while/LSTM/add_grad/tuple/control_dependencyIdentity)gradients/rnn/while/LSTM/add_grad/Reshape3^gradients/rnn/while/LSTM/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/rnn/while/LSTM/add_grad/Reshape
�
<gradients/rnn/while/LSTM/add_grad/tuple/control_dependency_1Identity+gradients/rnn/while/LSTM/add_grad/Reshape_13^gradients/rnn/while/LSTM/add_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/add_grad/Reshape_1
�
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIteration<gradients/rnn/while/LSTM/mul_grad/tuple/control_dependency_1*
T0
�
*gradients/rnn/while/LSTM/split_grad/concatConcatV23gradients/rnn/while/LSTM/Sigmoid_1_grad/SigmoidGrad+gradients/rnn/while/LSTM/Tanh_grad/TanhGrad:gradients/rnn/while/LSTM/add_grad/tuple/control_dependency3gradients/rnn/while/LSTM/Sigmoid_2_grad/SigmoidGrad0gradients/rnn/while/LSTM/split_grad/concat/Const*
N*

Tidx0*
T0
l
0gradients/rnn/while/LSTM/split_grad/concat/ConstConst^gradients/Sub_6*
dtype0*
value	B :
�
1gradients/rnn/while/LSTM/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/rnn/while/LSTM/split_grad/concat*
T0*
data_formatNHWC
�
6gradients/rnn/while/LSTM/BiasAdd_grad/tuple/group_depsNoOp2^gradients/rnn/while/LSTM/BiasAdd_grad/BiasAddGrad+^gradients/rnn/while/LSTM/split_grad/concat
�
>gradients/rnn/while/LSTM/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/rnn/while/LSTM/split_grad/concat7^gradients/rnn/while/LSTM/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/LSTM/split_grad/concat
�
@gradients/rnn/while/LSTM/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/rnn/while/LSTM/BiasAdd_grad/BiasAddGrad7^gradients/rnn/while/LSTM/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/while/LSTM/BiasAdd_grad/BiasAddGrad
�
+gradients/rnn/while/LSTM/MatMul_grad/MatMulMatMul>gradients/rnn/while/LSTM/BiasAdd_grad/tuple/control_dependency1gradients/rnn/while/LSTM/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( 
�
1gradients/rnn/while/LSTM/MatMul_grad/MatMul/EnterEnterrnn/LSTM/kernel/read*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
-gradients/rnn/while/LSTM/MatMul_grad/MatMul_1MatMul8gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPopV2>gradients/rnn/while/LSTM/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
3gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/ConstConst*
dtype0*(
_class
loc:@rnn/while/LSTM/concat*
valueB :
���������
�
3gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/f_accStackV23gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/Const*(
_class
loc:@rnn/while/LSTM/concat*

stack_name *
	elem_type0
�
3gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/EnterEnter3gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
9gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPushV2StackPushV23gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/Enterrnn/while/LSTM/concat^gradients/Add_6*
T0*
swap_memory( 
�
8gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPopV2
StackPopV2>gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
>gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/StackPopV2/EnterEnter3gradients/rnn/while/LSTM/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
5gradients/rnn/while/LSTM/MatMul_grad/tuple/group_depsNoOp,^gradients/rnn/while/LSTM/MatMul_grad/MatMul.^gradients/rnn/while/LSTM/MatMul_grad/MatMul_1
�
=gradients/rnn/while/LSTM/MatMul_grad/tuple/control_dependencyIdentity+gradients/rnn/while/LSTM/MatMul_grad/MatMul6^gradients/rnn/while/LSTM/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/rnn/while/LSTM/MatMul_grad/MatMul
�
?gradients/rnn/while/LSTM/MatMul_grad/tuple/control_dependency_1Identity-gradients/rnn/while/LSTM/MatMul_grad/MatMul_16^gradients/rnn/while/LSTM/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/rnn/while/LSTM/MatMul_grad/MatMul_1
c
1gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_1Enter1gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_2Merge3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_19gradients/rnn/while/LSTM/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
2gradients/rnn/while/LSTM/BiasAdd/Enter_grad/SwitchSwitch3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_2gradients/b_count_26*
T0
�
/gradients/rnn/while/LSTM/BiasAdd/Enter_grad/AddAdd4gradients/rnn/while/LSTM/BiasAdd/Enter_grad/Switch:1@gradients/rnn/while/LSTM/BiasAdd_grad/tuple/control_dependency_1*
T0
�
9gradients/rnn/while/LSTM/BiasAdd/Enter_grad/NextIterationNextIteration/gradients/rnn/while/LSTM/BiasAdd/Enter_grad/Add*
T0
x
3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_3Exit2gradients/rnn/while/LSTM/BiasAdd/Enter_grad/Switch*
T0
f
*gradients/rnn/while/LSTM/concat_grad/ConstConst^gradients/Sub_6*
value	B :*
dtype0
e
)gradients/rnn/while/LSTM/concat_grad/RankConst^gradients/Sub_6*
value	B :*
dtype0
�
(gradients/rnn/while/LSTM/concat_grad/modFloorMod*gradients/rnn/while/LSTM/concat_grad/Const)gradients/rnn/while/LSTM/concat_grad/Rank*
T0
i
*gradients/rnn/while/LSTM/concat_grad/ShapeShapernn/while/TensorArrayReadV3*
T0*
out_type0
�
+gradients/rnn/while/LSTM/concat_grad/ShapeNShapeN6gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV28gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N
�
1gradients/rnn/while/LSTM/concat_grad/ShapeN/ConstConst*
dtype0*.
_class$
" loc:@rnn/while/TensorArrayReadV3*
valueB :
���������
�
1gradients/rnn/while/LSTM/concat_grad/ShapeN/f_accStackV21gradients/rnn/while/LSTM/concat_grad/ShapeN/Const*

stack_name *
	elem_type0*.
_class$
" loc:@rnn/while/TensorArrayReadV3
�
1gradients/rnn/while/LSTM/concat_grad/ShapeN/EnterEnter1gradients/rnn/while/LSTM/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
7gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPushV2StackPushV21gradients/rnn/while/LSTM/concat_grad/ShapeN/Enterrnn/while/TensorArrayReadV3^gradients/Add_6*
T0*
swap_memory( 
�
6gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2
StackPopV2<gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_6*
	elem_type0
�
<gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2/EnterEnter1gradients/rnn/while/LSTM/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
3gradients/rnn/while/LSTM/concat_grad/ShapeN/Const_1Const*'
_class
loc:@rnn/while/Identity_4*
valueB :
���������*
dtype0
�
3gradients/rnn/while/LSTM/concat_grad/ShapeN/f_acc_1StackV23gradients/rnn/while/LSTM/concat_grad/ShapeN/Const_1*
	elem_type0*'
_class
loc:@rnn/while/Identity_4*

stack_name 
�
3gradients/rnn/while/LSTM/concat_grad/ShapeN/Enter_1Enter3gradients/rnn/while/LSTM/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
9gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPushV2_1StackPushV23gradients/rnn/while/LSTM/concat_grad/ShapeN/Enter_1rnn/while/Identity_4^gradients/Add_6*
T0*
swap_memory( 
�
8gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2_1
StackPopV2>gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub_6*
	elem_type0
�
>gradients/rnn/while/LSTM/concat_grad/ShapeN/StackPopV2_1/EnterEnter3gradients/rnn/while/LSTM/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
1gradients/rnn/while/LSTM/concat_grad/ConcatOffsetConcatOffset(gradients/rnn/while/LSTM/concat_grad/mod+gradients/rnn/while/LSTM/concat_grad/ShapeN-gradients/rnn/while/LSTM/concat_grad/ShapeN:1*
N
�
*gradients/rnn/while/LSTM/concat_grad/SliceSlice=gradients/rnn/while/LSTM/MatMul_grad/tuple/control_dependency1gradients/rnn/while/LSTM/concat_grad/ConcatOffset+gradients/rnn/while/LSTM/concat_grad/ShapeN*
T0*
Index0
�
,gradients/rnn/while/LSTM/concat_grad/Slice_1Slice=gradients/rnn/while/LSTM/MatMul_grad/tuple/control_dependency3gradients/rnn/while/LSTM/concat_grad/ConcatOffset:1-gradients/rnn/while/LSTM/concat_grad/ShapeN:1*
T0*
Index0
�
5gradients/rnn/while/LSTM/concat_grad/tuple/group_depsNoOp+^gradients/rnn/while/LSTM/concat_grad/Slice-^gradients/rnn/while/LSTM/concat_grad/Slice_1
�
=gradients/rnn/while/LSTM/concat_grad/tuple/control_dependencyIdentity*gradients/rnn/while/LSTM/concat_grad/Slice6^gradients/rnn/while/LSTM/concat_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/rnn/while/LSTM/concat_grad/Slice
�
?gradients/rnn/while/LSTM/concat_grad/tuple/control_dependency_1Identity,gradients/rnn/while/LSTM/concat_grad/Slice_16^gradients/rnn/while/LSTM/concat_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/rnn/while/LSTM/concat_grad/Slice_1
f
0gradients/rnn/while/LSTM/MatMul/Enter_grad/b_accConst*
valueB	~�*    *
dtype0
�
2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_1Enter0gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context
�
2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_2Merge2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_18gradients/rnn/while/LSTM/MatMul/Enter_grad/NextIteration*
T0*
N
�
1gradients/rnn/while/LSTM/MatMul/Enter_grad/SwitchSwitch2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_2gradients/b_count_26*
T0
�
.gradients/rnn/while/LSTM/MatMul/Enter_grad/AddAdd3gradients/rnn/while/LSTM/MatMul/Enter_grad/Switch:1?gradients/rnn/while/LSTM/MatMul_grad/tuple/control_dependency_1*
T0
�
8gradients/rnn/while/LSTM/MatMul/Enter_grad/NextIterationNextIteration.gradients/rnn/while/LSTM/MatMul/Enter_grad/Add*
T0
v
2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_3Exit1gradients/rnn/while/LSTM/MatMul/Enter_grad/Switch*
T0
�
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIteration?gradients/rnn/while/LSTM/concat_grad/tuple/control_dependency_1*
T0
q
beta1_power/initial_valueConst*)
_class
loc:@fully_connected/biases*
valueB
 *fff?*
dtype0
�
beta1_power
VariableV2*
shape: *
shared_name *)
_class
loc:@fully_connected/biases*
dtype0*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(
]
beta1_power/readIdentitybeta1_power*
T0*)
_class
loc:@fully_connected/biases
q
beta2_power/initial_valueConst*)
_class
loc:@fully_connected/biases*
valueB
 *w�?*
dtype0
�
beta2_power
VariableV2*
shared_name *)
_class
loc:@fully_connected/biases*
dtype0*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(
]
beta2_power/readIdentitybeta2_power*
T0*)
_class
loc:@fully_connected/biases
�
6rnn/LSTM/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"~   �  *"
_class
loc:@rnn/LSTM/kernel
}
,rnn/LSTM/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@rnn/LSTM/kernel*
dtype0
�
&rnn/LSTM/kernel/Adam/Initializer/zerosFill6rnn/LSTM/kernel/Adam/Initializer/zeros/shape_as_tensor,rnn/LSTM/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@rnn/LSTM/kernel
�
rnn/LSTM/kernel/Adam
VariableV2*
dtype0*
	container *
shape:	~�*
shared_name *"
_class
loc:@rnn/LSTM/kernel
�
rnn/LSTM/kernel/Adam/AssignAssignrnn/LSTM/kernel/Adam&rnn/LSTM/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel*
validate_shape(
h
rnn/LSTM/kernel/Adam/readIdentityrnn/LSTM/kernel/Adam*
T0*"
_class
loc:@rnn/LSTM/kernel
�
8rnn/LSTM/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"~   �  *"
_class
loc:@rnn/LSTM/kernel

.rnn/LSTM/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *"
_class
loc:@rnn/LSTM/kernel
�
(rnn/LSTM/kernel/Adam_1/Initializer/zerosFill8rnn/LSTM/kernel/Adam_1/Initializer/zeros/shape_as_tensor.rnn/LSTM/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@rnn/LSTM/kernel
�
rnn/LSTM/kernel/Adam_1
VariableV2*
shape:	~�*
shared_name *"
_class
loc:@rnn/LSTM/kernel*
dtype0*
	container 
�
rnn/LSTM/kernel/Adam_1/AssignAssignrnn/LSTM/kernel/Adam_1(rnn/LSTM/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel
l
rnn/LSTM/kernel/Adam_1/readIdentityrnn/LSTM/kernel/Adam_1*
T0*"
_class
loc:@rnn/LSTM/kernel
x
$rnn/LSTM/bias/Adam/Initializer/zerosConst*
valueB�*    * 
_class
loc:@rnn/LSTM/bias*
dtype0
�
rnn/LSTM/bias/Adam
VariableV2*
shared_name * 
_class
loc:@rnn/LSTM/bias*
dtype0*
	container *
shape:�
�
rnn/LSTM/bias/Adam/AssignAssignrnn/LSTM/bias/Adam$rnn/LSTM/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias*
validate_shape(
b
rnn/LSTM/bias/Adam/readIdentityrnn/LSTM/bias/Adam*
T0* 
_class
loc:@rnn/LSTM/bias
z
&rnn/LSTM/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    * 
_class
loc:@rnn/LSTM/bias
�
rnn/LSTM/bias/Adam_1
VariableV2*
shape:�*
shared_name * 
_class
loc:@rnn/LSTM/bias*
dtype0*
	container 
�
rnn/LSTM/bias/Adam_1/AssignAssignrnn/LSTM/bias/Adam_1&rnn/LSTM/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias*
validate_shape(
f
rnn/LSTM/bias/Adam_1/readIdentityrnn/LSTM/bias/Adam_1*
T0* 
_class
loc:@rnn/LSTM/bias
�
8rnn/LSTM_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_1/kernel*
dtype0
�
.rnn/LSTM_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *$
_class
loc:@rnn/LSTM_1/kernel
�
(rnn/LSTM_1/kernel/Adam/Initializer/zerosFill8rnn/LSTM_1/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_1/kernel
�
rnn/LSTM_1/kernel/Adam
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_1/kernel
�
rnn/LSTM_1/kernel/Adam/AssignAssignrnn/LSTM_1/kernel/Adam(rnn/LSTM_1/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_1/kernel
n
rnn/LSTM_1/kernel/Adam/readIdentityrnn/LSTM_1/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
:rnn/LSTM_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_1/kernel*
dtype0
�
0rnn/LSTM_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_1/kernel*
dtype0
�
*rnn/LSTM_1/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_1/kernel
�
rnn/LSTM_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_1/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_1/kernel/Adam_1/AssignAssignrnn/LSTM_1/kernel/Adam_1*rnn/LSTM_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_1/kernel*
validate_shape(
r
rnn/LSTM_1/kernel/Adam_1/readIdentityrnn/LSTM_1/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_1/kernel
|
&rnn/LSTM_1/bias/Adam/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_1/bias*
dtype0
�
rnn/LSTM_1/bias/Adam
VariableV2*
dtype0*
	container *
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_1/bias
�
rnn/LSTM_1/bias/Adam/AssignAssignrnn/LSTM_1/bias/Adam&rnn/LSTM_1/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(*
use_locking(
h
rnn/LSTM_1/bias/Adam/readIdentityrnn/LSTM_1/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_1/bias
~
(rnn/LSTM_1/bias/Adam_1/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_1/bias*
dtype0
�
rnn/LSTM_1/bias/Adam_1
VariableV2*"
_class
loc:@rnn/LSTM_1/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_1/bias/Adam_1/AssignAssignrnn/LSTM_1/bias/Adam_1(rnn/LSTM_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(
l
rnn/LSTM_1/bias/Adam_1/readIdentityrnn/LSTM_1/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_1/bias
�
8rnn/LSTM_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_2/kernel*
dtype0
�
.rnn/LSTM_2/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_2/kernel*
dtype0
�
(rnn/LSTM_2/kernel/Adam/Initializer/zerosFill8rnn/LSTM_2/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_2/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_2/kernel
�
rnn/LSTM_2/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_2/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_2/kernel/Adam/AssignAssignrnn/LSTM_2/kernel/Adam(rnn/LSTM_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
validate_shape(*
use_locking(
n
rnn/LSTM_2/kernel/Adam/readIdentityrnn/LSTM_2/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_2/kernel
�
:rnn/LSTM_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"�   �  *$
_class
loc:@rnn/LSTM_2/kernel
�
0rnn/LSTM_2/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_2/kernel*
dtype0
�
*rnn/LSTM_2/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_2/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_2/kernel
�
rnn/LSTM_2/kernel/Adam_1
VariableV2*
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_2/kernel*
dtype0*
	container 
�
rnn/LSTM_2/kernel/Adam_1/AssignAssignrnn/LSTM_2/kernel/Adam_1*rnn/LSTM_2/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
validate_shape(*
use_locking(
r
rnn/LSTM_2/kernel/Adam_1/readIdentityrnn/LSTM_2/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_2/kernel
|
&rnn/LSTM_2/bias/Adam/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_2/bias*
dtype0
�
rnn/LSTM_2/bias/Adam
VariableV2*"
_class
loc:@rnn/LSTM_2/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_2/bias/Adam/AssignAssignrnn/LSTM_2/bias/Adam&rnn/LSTM_2/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_2/bias
h
rnn/LSTM_2/bias/Adam/readIdentityrnn/LSTM_2/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_2/bias
~
(rnn/LSTM_2/bias/Adam_1/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_2/bias*
dtype0
�
rnn/LSTM_2/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@rnn/LSTM_2/bias*
dtype0*
	container *
shape:�
�
rnn/LSTM_2/bias/Adam_1/AssignAssignrnn/LSTM_2/bias/Adam_1(rnn/LSTM_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_2/bias*
validate_shape(
l
rnn/LSTM_2/bias/Adam_1/readIdentityrnn/LSTM_2/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_2/bias
�
8rnn/LSTM_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_3/kernel*
dtype0
�
.rnn/LSTM_3/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_3/kernel*
dtype0
�
(rnn/LSTM_3/kernel/Adam/Initializer/zerosFill8rnn/LSTM_3/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_3/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_3/kernel
�
rnn/LSTM_3/kernel/Adam
VariableV2*$
_class
loc:@rnn/LSTM_3/kernel*
dtype0*
	container *
shape:
��*
shared_name 
�
rnn/LSTM_3/kernel/Adam/AssignAssignrnn/LSTM_3/kernel/Adam(rnn/LSTM_3/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_3/kernel
n
rnn/LSTM_3/kernel/Adam/readIdentityrnn/LSTM_3/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
:rnn/LSTM_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_3/kernel*
dtype0
�
0rnn/LSTM_3/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *$
_class
loc:@rnn/LSTM_3/kernel
�
*rnn/LSTM_3/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_3/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_3/kernel
�
rnn/LSTM_3/kernel/Adam_1
VariableV2*$
_class
loc:@rnn/LSTM_3/kernel*
dtype0*
	container *
shape:
��*
shared_name 
�
rnn/LSTM_3/kernel/Adam_1/AssignAssignrnn/LSTM_3/kernel/Adam_1*rnn/LSTM_3/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@rnn/LSTM_3/kernel*
validate_shape(*
use_locking(
r
rnn/LSTM_3/kernel/Adam_1/readIdentityrnn/LSTM_3/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_3/kernel
|
&rnn/LSTM_3/bias/Adam/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_3/bias*
dtype0
�
rnn/LSTM_3/bias/Adam
VariableV2*
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_3/bias*
dtype0*
	container 
�
rnn/LSTM_3/bias/Adam/AssignAssignrnn/LSTM_3/bias/Adam&rnn/LSTM_3/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@rnn/LSTM_3/bias*
validate_shape(*
use_locking(
h
rnn/LSTM_3/bias/Adam/readIdentityrnn/LSTM_3/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_3/bias
~
(rnn/LSTM_3/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *"
_class
loc:@rnn/LSTM_3/bias
�
rnn/LSTM_3/bias/Adam_1
VariableV2*"
_class
loc:@rnn/LSTM_3/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_3/bias/Adam_1/AssignAssignrnn/LSTM_3/bias/Adam_1(rnn/LSTM_3/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_3/bias
l
rnn/LSTM_3/bias/Adam_1/readIdentityrnn/LSTM_3/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_3/bias
�
8rnn/LSTM_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"�   �  *$
_class
loc:@rnn/LSTM_4/kernel
�
.rnn/LSTM_4/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_4/kernel*
dtype0
�
(rnn/LSTM_4/kernel/Adam/Initializer/zerosFill8rnn/LSTM_4/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_4/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_4/kernel
�
rnn/LSTM_4/kernel/Adam
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_4/kernel
�
rnn/LSTM_4/kernel/Adam/AssignAssignrnn/LSTM_4/kernel/Adam(rnn/LSTM_4/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
validate_shape(
n
rnn/LSTM_4/kernel/Adam/readIdentityrnn/LSTM_4/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
:rnn/LSTM_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_4/kernel*
dtype0
�
0rnn/LSTM_4/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_4/kernel*
dtype0
�
*rnn/LSTM_4/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_4/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_4/kernel
�
rnn/LSTM_4/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_4/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_4/kernel/Adam_1/AssignAssignrnn/LSTM_4/kernel/Adam_1*rnn/LSTM_4/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
validate_shape(
r
rnn/LSTM_4/kernel/Adam_1/readIdentityrnn/LSTM_4/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_4/kernel
|
&rnn/LSTM_4/bias/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *"
_class
loc:@rnn/LSTM_4/bias
�
rnn/LSTM_4/bias/Adam
VariableV2*
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_4/bias*
dtype0*
	container 
�
rnn/LSTM_4/bias/Adam/AssignAssignrnn/LSTM_4/bias/Adam&rnn/LSTM_4/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_4/bias
h
rnn/LSTM_4/bias/Adam/readIdentityrnn/LSTM_4/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_4/bias
~
(rnn/LSTM_4/bias/Adam_1/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_4/bias*
dtype0
�
rnn/LSTM_4/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@rnn/LSTM_4/bias*
dtype0*
	container *
shape:�
�
rnn/LSTM_4/bias/Adam_1/AssignAssignrnn/LSTM_4/bias/Adam_1(rnn/LSTM_4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_4/bias*
validate_shape(
l
rnn/LSTM_4/bias/Adam_1/readIdentityrnn/LSTM_4/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_4/bias
�
8rnn/LSTM_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"�   �  *$
_class
loc:@rnn/LSTM_5/kernel
�
.rnn/LSTM_5/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_5/kernel*
dtype0
�
(rnn/LSTM_5/kernel/Adam/Initializer/zerosFill8rnn/LSTM_5/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_5/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_5/kernel
�
rnn/LSTM_5/kernel/Adam
VariableV2*
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_5/kernel*
dtype0*
	container 
�
rnn/LSTM_5/kernel/Adam/AssignAssignrnn/LSTM_5/kernel/Adam(rnn/LSTM_5/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(
n
rnn/LSTM_5/kernel/Adam/readIdentityrnn/LSTM_5/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
:rnn/LSTM_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"�   �  *$
_class
loc:@rnn/LSTM_5/kernel
�
0rnn/LSTM_5/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_5/kernel*
dtype0
�
*rnn/LSTM_5/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_5/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_5/kernel
�
rnn/LSTM_5/kernel/Adam_1
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *$
_class
loc:@rnn/LSTM_5/kernel
�
rnn/LSTM_5/kernel/Adam_1/AssignAssignrnn/LSTM_5/kernel/Adam_1*rnn/LSTM_5/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(
r
rnn/LSTM_5/kernel/Adam_1/readIdentityrnn/LSTM_5/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_5/kernel
|
&rnn/LSTM_5/bias/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *"
_class
loc:@rnn/LSTM_5/bias
�
rnn/LSTM_5/bias/Adam
VariableV2*
shared_name *"
_class
loc:@rnn/LSTM_5/bias*
dtype0*
	container *
shape:�
�
rnn/LSTM_5/bias/Adam/AssignAssignrnn/LSTM_5/bias/Adam&rnn/LSTM_5/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias*
validate_shape(
h
rnn/LSTM_5/bias/Adam/readIdentityrnn/LSTM_5/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_5/bias
~
(rnn/LSTM_5/bias/Adam_1/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_5/bias*
dtype0
�
rnn/LSTM_5/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *"
_class
loc:@rnn/LSTM_5/bias
�
rnn/LSTM_5/bias/Adam_1/AssignAssignrnn/LSTM_5/bias/Adam_1(rnn/LSTM_5/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias*
validate_shape(
l
rnn/LSTM_5/bias/Adam_1/readIdentityrnn/LSTM_5/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_5/bias
�
8rnn/LSTM_6/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"�   �  *$
_class
loc:@rnn/LSTM_6/kernel
�
.rnn/LSTM_6/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_6/kernel*
dtype0
�
(rnn/LSTM_6/kernel/Adam/Initializer/zerosFill8rnn/LSTM_6/kernel/Adam/Initializer/zeros/shape_as_tensor.rnn/LSTM_6/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_6/kernel
�
rnn/LSTM_6/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@rnn/LSTM_6/kernel*
dtype0*
	container *
shape:
��
�
rnn/LSTM_6/kernel/Adam/AssignAssignrnn/LSTM_6/kernel/Adam(rnn/LSTM_6/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_6/kernel*
validate_shape(
n
rnn/LSTM_6/kernel/Adam/readIdentityrnn/LSTM_6/kernel/Adam*
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
:rnn/LSTM_6/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�   �  *$
_class
loc:@rnn/LSTM_6/kernel*
dtype0
�
0rnn/LSTM_6/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@rnn/LSTM_6/kernel*
dtype0
�
*rnn/LSTM_6/kernel/Adam_1/Initializer/zerosFill:rnn/LSTM_6/kernel/Adam_1/Initializer/zeros/shape_as_tensor0rnn/LSTM_6/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@rnn/LSTM_6/kernel
�
rnn/LSTM_6/kernel/Adam_1
VariableV2*$
_class
loc:@rnn/LSTM_6/kernel*
dtype0*
	container *
shape:
��*
shared_name 
�
rnn/LSTM_6/kernel/Adam_1/AssignAssignrnn/LSTM_6/kernel/Adam_1*rnn/LSTM_6/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_6/kernel
r
rnn/LSTM_6/kernel/Adam_1/readIdentityrnn/LSTM_6/kernel/Adam_1*
T0*$
_class
loc:@rnn/LSTM_6/kernel
|
&rnn/LSTM_6/bias/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *"
_class
loc:@rnn/LSTM_6/bias
�
rnn/LSTM_6/bias/Adam
VariableV2*"
_class
loc:@rnn/LSTM_6/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_6/bias/Adam/AssignAssignrnn/LSTM_6/bias/Adam&rnn/LSTM_6/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_6/bias*
validate_shape(
h
rnn/LSTM_6/bias/Adam/readIdentityrnn/LSTM_6/bias/Adam*
T0*"
_class
loc:@rnn/LSTM_6/bias
~
(rnn/LSTM_6/bias/Adam_1/Initializer/zerosConst*
valueB�*    *"
_class
loc:@rnn/LSTM_6/bias*
dtype0
�
rnn/LSTM_6/bias/Adam_1
VariableV2*"
_class
loc:@rnn/LSTM_6/bias*
dtype0*
	container *
shape:�*
shared_name 
�
rnn/LSTM_6/bias/Adam_1/AssignAssignrnn/LSTM_6/bias/Adam_1(rnn/LSTM_6/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_6/bias*
validate_shape(
l
rnn/LSTM_6/bias/Adam_1/readIdentityrnn/LSTM_6/bias/Adam_1*
T0*"
_class
loc:@rnn/LSTM_6/bias
�
.fully_connected/weights/Adam/Initializer/zerosConst*
valueBx*    **
_class 
loc:@fully_connected/weights*
dtype0
�
fully_connected/weights/Adam
VariableV2*
shared_name **
_class 
loc:@fully_connected/weights*
dtype0*
	container *
shape
:x
�
#fully_connected/weights/Adam/AssignAssignfully_connected/weights/Adam.fully_connected/weights/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(
�
!fully_connected/weights/Adam/readIdentityfully_connected/weights/Adam*
T0**
_class 
loc:@fully_connected/weights
�
0fully_connected/weights/Adam_1/Initializer/zerosConst*
valueBx*    **
_class 
loc:@fully_connected/weights*
dtype0
�
fully_connected/weights/Adam_1
VariableV2*
shape
:x*
shared_name **
_class 
loc:@fully_connected/weights*
dtype0*
	container 
�
%fully_connected/weights/Adam_1/AssignAssignfully_connected/weights/Adam_10fully_connected/weights/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(
�
#fully_connected/weights/Adam_1/readIdentityfully_connected/weights/Adam_1*
T0**
_class 
loc:@fully_connected/weights
�
-fully_connected/biases/Adam/Initializer/zerosConst*
valueB*    *)
_class
loc:@fully_connected/biases*
dtype0
�
fully_connected/biases/Adam
VariableV2*
shape:*
shared_name *)
_class
loc:@fully_connected/biases*
dtype0*
	container 
�
"fully_connected/biases/Adam/AssignAssignfully_connected/biases/Adam-fully_connected/biases/Adam/Initializer/zeros*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
use_locking(
}
 fully_connected/biases/Adam/readIdentityfully_connected/biases/Adam*
T0*)
_class
loc:@fully_connected/biases
�
/fully_connected/biases/Adam_1/Initializer/zerosConst*
valueB*    *)
_class
loc:@fully_connected/biases*
dtype0
�
fully_connected/biases/Adam_1
VariableV2*)
_class
loc:@fully_connected/biases*
dtype0*
	container *
shape:*
shared_name 
�
$fully_connected/biases/Adam_1/AssignAssignfully_connected/biases/Adam_1/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
�
"fully_connected/biases/Adam_1/readIdentityfully_connected/biases/Adam_1*
T0*)
_class
loc:@fully_connected/biases
?
Adam/learning_rateConst*
valueB
 *
�#<*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w�?*
dtype0
9
Adam/epsilonConst*
valueB
 *w�+2*
dtype0
�
%Adam/update_rnn/LSTM/kernel/ApplyAdam	ApplyAdamrnn/LSTM/kernelrnn/LSTM/kernel/Adamrnn/LSTM/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/rnn/while/LSTM/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*"
_class
loc:@rnn/LSTM/kernel*
use_nesterov( 
�
#Adam/update_rnn/LSTM/bias/ApplyAdam	ApplyAdamrnn/LSTM/biasrnn/LSTM/bias/Adamrnn/LSTM/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/rnn/while/LSTM/BiasAdd/Enter_grad/b_acc_3*
T0* 
_class
loc:@rnn/LSTM/bias*
use_nesterov( *
use_locking( 
�
'Adam/update_rnn/LSTM_1/kernel/ApplyAdam	ApplyAdamrnn/LSTM_1/kernelrnn/LSTM_1/kernel/Adamrnn/LSTM_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_1/while/LSTM_1/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*$
_class
loc:@rnn/LSTM_1/kernel
�
%Adam/update_rnn/LSTM_1/bias/ApplyAdam	ApplyAdamrnn/LSTM_1/biasrnn/LSTM_1/bias/Adamrnn/LSTM_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_1/while/LSTM_1/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*"
_class
loc:@rnn/LSTM_1/bias*
use_nesterov( 
�
'Adam/update_rnn/LSTM_2/kernel/ApplyAdam	ApplyAdamrnn/LSTM_2/kernelrnn/LSTM_2/kernel/Adamrnn/LSTM_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_2/while/LSTM_2/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*$
_class
loc:@rnn/LSTM_2/kernel*
use_nesterov( 
�
%Adam/update_rnn/LSTM_2/bias/ApplyAdam	ApplyAdamrnn/LSTM_2/biasrnn/LSTM_2/bias/Adamrnn/LSTM_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_2/while/LSTM_2/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*"
_class
loc:@rnn/LSTM_2/bias
�
'Adam/update_rnn/LSTM_3/kernel/ApplyAdam	ApplyAdamrnn/LSTM_3/kernelrnn/LSTM_3/kernel/Adamrnn/LSTM_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_3/while/LSTM_3/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
%Adam/update_rnn/LSTM_3/bias/ApplyAdam	ApplyAdamrnn/LSTM_3/biasrnn/LSTM_3/bias/Adamrnn/LSTM_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_3/while/LSTM_3/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*"
_class
loc:@rnn/LSTM_3/bias*
use_nesterov( 
�
'Adam/update_rnn/LSTM_4/kernel/ApplyAdam	ApplyAdamrnn/LSTM_4/kernelrnn/LSTM_4/kernel/Adamrnn/LSTM_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_4/while/LSTM_4/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
%Adam/update_rnn/LSTM_4/bias/ApplyAdam	ApplyAdamrnn/LSTM_4/biasrnn/LSTM_4/bias/Adamrnn/LSTM_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_4/while/LSTM_4/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*"
_class
loc:@rnn/LSTM_4/bias*
use_nesterov( 
�
'Adam/update_rnn/LSTM_5/kernel/ApplyAdam	ApplyAdamrnn/LSTM_5/kernelrnn/LSTM_5/kernel/Adamrnn/LSTM_5/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_5/while/LSTM_5/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*$
_class
loc:@rnn/LSTM_5/kernel
�
%Adam/update_rnn/LSTM_5/bias/ApplyAdam	ApplyAdamrnn/LSTM_5/biasrnn/LSTM_5/bias/Adamrnn/LSTM_5/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_5/while/LSTM_5/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*"
_class
loc:@rnn/LSTM_5/bias
�
'Adam/update_rnn/LSTM_6/kernel/ApplyAdam	ApplyAdamrnn/LSTM_6/kernelrnn/LSTM_6/kernel/Adamrnn/LSTM_6/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/rnn_6/while/LSTM_6/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
use_locking( *
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
%Adam/update_rnn/LSTM_6/bias/ApplyAdam	ApplyAdamrnn/LSTM_6/biasrnn/LSTM_6/bias/Adamrnn/LSTM_6/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/rnn_6/while/LSTM_6/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*"
_class
loc:@rnn/LSTM_6/bias*
use_nesterov( 
�
-Adam/update_fully_connected/weights/ApplyAdam	ApplyAdamfully_connected/weightsfully_connected/weights/Adamfully_connected/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@fully_connected/weights*
use_nesterov( 
�
,Adam/update_fully_connected/biases/ApplyAdam	ApplyAdamfully_connected/biasesfully_connected/biases/Adamfully_connected/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*)
_class
loc:@fully_connected/biases
�
Adam/mulMulbeta1_power/read
Adam/beta1-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam$^Adam/update_rnn/LSTM/bias/ApplyAdam&^Adam/update_rnn/LSTM/kernel/ApplyAdam&^Adam/update_rnn/LSTM_1/bias/ApplyAdam(^Adam/update_rnn/LSTM_1/kernel/ApplyAdam&^Adam/update_rnn/LSTM_2/bias/ApplyAdam(^Adam/update_rnn/LSTM_2/kernel/ApplyAdam&^Adam/update_rnn/LSTM_3/bias/ApplyAdam(^Adam/update_rnn/LSTM_3/kernel/ApplyAdam&^Adam/update_rnn/LSTM_4/bias/ApplyAdam(^Adam/update_rnn/LSTM_4/kernel/ApplyAdam&^Adam/update_rnn/LSTM_5/bias/ApplyAdam(^Adam/update_rnn/LSTM_5/kernel/ApplyAdam&^Adam/update_rnn/LSTM_6/bias/ApplyAdam(^Adam/update_rnn/LSTM_6/kernel/ApplyAdam*
T0*)
_class
loc:@fully_connected/biases
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*)
_class
loc:@fully_connected/biases*
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam$^Adam/update_rnn/LSTM/bias/ApplyAdam&^Adam/update_rnn/LSTM/kernel/ApplyAdam&^Adam/update_rnn/LSTM_1/bias/ApplyAdam(^Adam/update_rnn/LSTM_1/kernel/ApplyAdam&^Adam/update_rnn/LSTM_2/bias/ApplyAdam(^Adam/update_rnn/LSTM_2/kernel/ApplyAdam&^Adam/update_rnn/LSTM_3/bias/ApplyAdam(^Adam/update_rnn/LSTM_3/kernel/ApplyAdam&^Adam/update_rnn/LSTM_4/bias/ApplyAdam(^Adam/update_rnn/LSTM_4/kernel/ApplyAdam&^Adam/update_rnn/LSTM_5/bias/ApplyAdam(^Adam/update_rnn/LSTM_5/kernel/ApplyAdam&^Adam/update_rnn/LSTM_6/bias/ApplyAdam(^Adam/update_rnn/LSTM_6/kernel/ApplyAdam*
T0*)
_class
loc:@fully_connected/biases
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
use_locking( 
�
AdamNoOp^Adam/Assign^Adam/Assign_1-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam$^Adam/update_rnn/LSTM/bias/ApplyAdam&^Adam/update_rnn/LSTM/kernel/ApplyAdam&^Adam/update_rnn/LSTM_1/bias/ApplyAdam(^Adam/update_rnn/LSTM_1/kernel/ApplyAdam&^Adam/update_rnn/LSTM_2/bias/ApplyAdam(^Adam/update_rnn/LSTM_2/kernel/ApplyAdam&^Adam/update_rnn/LSTM_3/bias/ApplyAdam(^Adam/update_rnn/LSTM_3/kernel/ApplyAdam&^Adam/update_rnn/LSTM_4/bias/ApplyAdam(^Adam/update_rnn/LSTM_4/kernel/ApplyAdam&^Adam/update_rnn/LSTM_5/bias/ApplyAdam(^Adam/update_rnn/LSTM_5/kernel/ApplyAdam&^Adam/update_rnn/LSTM_6/bias/ApplyAdam(^Adam/update_rnn/LSTM_6/kernel/ApplyAdam
M
SquaredDifferenceSquaredDifferencePlaceholder_2Placeholder_3*
T0
<
Const_1Const*
valueB"       *
dtype0
N
MeanMeanSquaredDifferenceConst_1*
T0*

Tidx0*
	keep_dims( 

SqrtSqrtMean*
T0
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
�	
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBfully_connected/biasesBfully_connected/biases/AdamBfully_connected/biases/Adam_1Bfully_connected/weightsBfully_connected/weights/AdamBfully_connected/weights/Adam_1Brnn/LSTM/biasBrnn/LSTM/bias/AdamBrnn/LSTM/bias/Adam_1Brnn/LSTM/kernelBrnn/LSTM/kernel/AdamBrnn/LSTM/kernel/Adam_1Brnn/LSTM_1/biasBrnn/LSTM_1/bias/AdamBrnn/LSTM_1/bias/Adam_1Brnn/LSTM_1/kernelBrnn/LSTM_1/kernel/AdamBrnn/LSTM_1/kernel/Adam_1Brnn/LSTM_2/biasBrnn/LSTM_2/bias/AdamBrnn/LSTM_2/bias/Adam_1Brnn/LSTM_2/kernelBrnn/LSTM_2/kernel/AdamBrnn/LSTM_2/kernel/Adam_1Brnn/LSTM_3/biasBrnn/LSTM_3/bias/AdamBrnn/LSTM_3/bias/Adam_1Brnn/LSTM_3/kernelBrnn/LSTM_3/kernel/AdamBrnn/LSTM_3/kernel/Adam_1Brnn/LSTM_4/biasBrnn/LSTM_4/bias/AdamBrnn/LSTM_4/bias/Adam_1Brnn/LSTM_4/kernelBrnn/LSTM_4/kernel/AdamBrnn/LSTM_4/kernel/Adam_1Brnn/LSTM_5/biasBrnn/LSTM_5/bias/AdamBrnn/LSTM_5/bias/Adam_1Brnn/LSTM_5/kernelBrnn/LSTM_5/kernel/AdamBrnn/LSTM_5/kernel/Adam_1Brnn/LSTM_6/biasBrnn/LSTM_6/bias/AdamBrnn/LSTM_6/bias/Adam_1Brnn/LSTM_6/kernelBrnn/LSTM_6/kernel/AdamBrnn/LSTM_6/kernel/Adam_1
�
save/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�	
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerfully_connected/biasesfully_connected/biases/Adamfully_connected/biases/Adam_1fully_connected/weightsfully_connected/weights/Adamfully_connected/weights/Adam_1rnn/LSTM/biasrnn/LSTM/bias/Adamrnn/LSTM/bias/Adam_1rnn/LSTM/kernelrnn/LSTM/kernel/Adamrnn/LSTM/kernel/Adam_1rnn/LSTM_1/biasrnn/LSTM_1/bias/Adamrnn/LSTM_1/bias/Adam_1rnn/LSTM_1/kernelrnn/LSTM_1/kernel/Adamrnn/LSTM_1/kernel/Adam_1rnn/LSTM_2/biasrnn/LSTM_2/bias/Adamrnn/LSTM_2/bias/Adam_1rnn/LSTM_2/kernelrnn/LSTM_2/kernel/Adamrnn/LSTM_2/kernel/Adam_1rnn/LSTM_3/biasrnn/LSTM_3/bias/Adamrnn/LSTM_3/bias/Adam_1rnn/LSTM_3/kernelrnn/LSTM_3/kernel/Adamrnn/LSTM_3/kernel/Adam_1rnn/LSTM_4/biasrnn/LSTM_4/bias/Adamrnn/LSTM_4/bias/Adam_1rnn/LSTM_4/kernelrnn/LSTM_4/kernel/Adamrnn/LSTM_4/kernel/Adam_1rnn/LSTM_5/biasrnn/LSTM_5/bias/Adamrnn/LSTM_5/bias/Adam_1rnn/LSTM_5/kernelrnn/LSTM_5/kernel/Adamrnn/LSTM_5/kernel/Adam_1rnn/LSTM_6/biasrnn/LSTM_6/bias/Adamrnn/LSTM_6/bias/Adam_1rnn/LSTM_6/kernelrnn/LSTM_6/kernel/Adamrnn/LSTM_6/kernel/Adam_1*@
dtypes6
422
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
�	
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�2Bbeta1_powerBbeta2_powerBfully_connected/biasesBfully_connected/biases/AdamBfully_connected/biases/Adam_1Bfully_connected/weightsBfully_connected/weights/AdamBfully_connected/weights/Adam_1Brnn/LSTM/biasBrnn/LSTM/bias/AdamBrnn/LSTM/bias/Adam_1Brnn/LSTM/kernelBrnn/LSTM/kernel/AdamBrnn/LSTM/kernel/Adam_1Brnn/LSTM_1/biasBrnn/LSTM_1/bias/AdamBrnn/LSTM_1/bias/Adam_1Brnn/LSTM_1/kernelBrnn/LSTM_1/kernel/AdamBrnn/LSTM_1/kernel/Adam_1Brnn/LSTM_2/biasBrnn/LSTM_2/bias/AdamBrnn/LSTM_2/bias/Adam_1Brnn/LSTM_2/kernelBrnn/LSTM_2/kernel/AdamBrnn/LSTM_2/kernel/Adam_1Brnn/LSTM_3/biasBrnn/LSTM_3/bias/AdamBrnn/LSTM_3/bias/Adam_1Brnn/LSTM_3/kernelBrnn/LSTM_3/kernel/AdamBrnn/LSTM_3/kernel/Adam_1Brnn/LSTM_4/biasBrnn/LSTM_4/bias/AdamBrnn/LSTM_4/bias/Adam_1Brnn/LSTM_4/kernelBrnn/LSTM_4/kernel/AdamBrnn/LSTM_4/kernel/Adam_1Brnn/LSTM_5/biasBrnn/LSTM_5/bias/AdamBrnn/LSTM_5/bias/Adam_1Brnn/LSTM_5/kernelBrnn/LSTM_5/kernel/AdamBrnn/LSTM_5/kernel/Adam_1Brnn/LSTM_6/biasBrnn/LSTM_6/bias/AdamBrnn/LSTM_6/bias/Adam_1Brnn/LSTM_6/kernelBrnn/LSTM_6/kernel/AdamBrnn/LSTM_6/kernel/Adam_1*
dtype0
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*@
dtypes6
422
�
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
�
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
�
save/Assign_2Assignfully_connected/biasessave/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(
�
save/Assign_3Assignfully_connected/biases/Adamsave/RestoreV2:3*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
use_locking(
�
save/Assign_4Assignfully_connected/biases/Adam_1save/RestoreV2:4*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
use_locking(
�
save/Assign_5Assignfully_connected/weightssave/RestoreV2:5*
validate_shape(*
use_locking(*
T0**
_class 
loc:@fully_connected/weights
�
save/Assign_6Assignfully_connected/weights/Adamsave/RestoreV2:6*
validate_shape(*
use_locking(*
T0**
_class 
loc:@fully_connected/weights
�
save/Assign_7Assignfully_connected/weights/Adam_1save/RestoreV2:7*
validate_shape(*
use_locking(*
T0**
_class 
loc:@fully_connected/weights
�
save/Assign_8Assignrnn/LSTM/biassave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias*
validate_shape(
�
save/Assign_9Assignrnn/LSTM/bias/Adamsave/RestoreV2:9*
validate_shape(*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias
�
save/Assign_10Assignrnn/LSTM/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0* 
_class
loc:@rnn/LSTM/bias*
validate_shape(
�
save/Assign_11Assignrnn/LSTM/kernelsave/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel*
validate_shape(
�
save/Assign_12Assignrnn/LSTM/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel*
validate_shape(
�
save/Assign_13Assignrnn/LSTM/kernel/Adam_1save/RestoreV2:13*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM/kernel
�
save/Assign_14Assignrnn/LSTM_1/biassave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(
�
save/Assign_15Assignrnn/LSTM_1/bias/Adamsave/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(
�
save/Assign_16Assignrnn/LSTM_1/bias/Adam_1save/RestoreV2:16*
T0*"
_class
loc:@rnn/LSTM_1/bias*
validate_shape(*
use_locking(
�
save/Assign_17Assignrnn/LSTM_1/kernelsave/RestoreV2:17*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_1/kernel*
validate_shape(
�
save/Assign_18Assignrnn/LSTM_1/kernel/Adamsave/RestoreV2:18*
T0*$
_class
loc:@rnn/LSTM_1/kernel*
validate_shape(*
use_locking(
�
save/Assign_19Assignrnn/LSTM_1/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_1/kernel*
validate_shape(
�
save/Assign_20Assignrnn/LSTM_2/biassave/RestoreV2:20*
T0*"
_class
loc:@rnn/LSTM_2/bias*
validate_shape(*
use_locking(
�
save/Assign_21Assignrnn/LSTM_2/bias/Adamsave/RestoreV2:21*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_2/bias*
validate_shape(
�
save/Assign_22Assignrnn/LSTM_2/bias/Adam_1save/RestoreV2:22*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_2/bias
�
save/Assign_23Assignrnn/LSTM_2/kernelsave/RestoreV2:23*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
validate_shape(*
use_locking(
�
save/Assign_24Assignrnn/LSTM_2/kernel/Adamsave/RestoreV2:24*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
validate_shape(*
use_locking(
�
save/Assign_25Assignrnn/LSTM_2/kernel/Adam_1save/RestoreV2:25*
T0*$
_class
loc:@rnn/LSTM_2/kernel*
validate_shape(*
use_locking(
�
save/Assign_26Assignrnn/LSTM_3/biassave/RestoreV2:26*
T0*"
_class
loc:@rnn/LSTM_3/bias*
validate_shape(*
use_locking(
�
save/Assign_27Assignrnn/LSTM_3/bias/Adamsave/RestoreV2:27*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_3/bias*
validate_shape(
�
save/Assign_28Assignrnn/LSTM_3/bias/Adam_1save/RestoreV2:28*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_3/bias*
validate_shape(
�
save/Assign_29Assignrnn/LSTM_3/kernelsave/RestoreV2:29*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_3/kernel
�
save/Assign_30Assignrnn/LSTM_3/kernel/Adamsave/RestoreV2:30*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_3/kernel*
validate_shape(
�
save/Assign_31Assignrnn/LSTM_3/kernel/Adam_1save/RestoreV2:31*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_3/kernel*
validate_shape(
�
save/Assign_32Assignrnn/LSTM_4/biassave/RestoreV2:32*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_4/bias
�
save/Assign_33Assignrnn/LSTM_4/bias/Adamsave/RestoreV2:33*
T0*"
_class
loc:@rnn/LSTM_4/bias*
validate_shape(*
use_locking(
�
save/Assign_34Assignrnn/LSTM_4/bias/Adam_1save/RestoreV2:34*
T0*"
_class
loc:@rnn/LSTM_4/bias*
validate_shape(*
use_locking(
�
save/Assign_35Assignrnn/LSTM_4/kernelsave/RestoreV2:35*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
validate_shape(
�
save/Assign_36Assignrnn/LSTM_4/kernel/Adamsave/RestoreV2:36*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel*
validate_shape(
�
save/Assign_37Assignrnn/LSTM_4/kernel/Adam_1save/RestoreV2:37*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_4/kernel
�
save/Assign_38Assignrnn/LSTM_5/biassave/RestoreV2:38*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias*
validate_shape(
�
save/Assign_39Assignrnn/LSTM_5/bias/Adamsave/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias*
validate_shape(
�
save/Assign_40Assignrnn/LSTM_5/bias/Adam_1save/RestoreV2:40*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_5/bias*
validate_shape(
�
save/Assign_41Assignrnn/LSTM_5/kernelsave/RestoreV2:41*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(*
use_locking(
�
save/Assign_42Assignrnn/LSTM_5/kernel/Adamsave/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(
�
save/Assign_43Assignrnn/LSTM_5/kernel/Adam_1save/RestoreV2:43*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_5/kernel*
validate_shape(
�
save/Assign_44Assignrnn/LSTM_6/biassave/RestoreV2:44*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_6/bias*
validate_shape(
�
save/Assign_45Assignrnn/LSTM_6/bias/Adamsave/RestoreV2:45*
validate_shape(*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_6/bias
�
save/Assign_46Assignrnn/LSTM_6/bias/Adam_1save/RestoreV2:46*
use_locking(*
T0*"
_class
loc:@rnn/LSTM_6/bias*
validate_shape(
�
save/Assign_47Assignrnn/LSTM_6/kernelsave/RestoreV2:47*
T0*$
_class
loc:@rnn/LSTM_6/kernel*
validate_shape(*
use_locking(
�
save/Assign_48Assignrnn/LSTM_6/kernel/Adamsave/RestoreV2:48*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_6/kernel*
validate_shape(
�
save/Assign_49Assignrnn/LSTM_6/kernel/Adam_1save/RestoreV2:49*
validate_shape(*
use_locking(*
T0*$
_class
loc:@rnn/LSTM_6/kernel
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^beta1_power/Assign^beta2_power/Assign#^fully_connected/biases/Adam/Assign%^fully_connected/biases/Adam_1/Assign^fully_connected/biases/Assign$^fully_connected/weights/Adam/Assign&^fully_connected/weights/Adam_1/Assign^fully_connected/weights/Assign^rnn/LSTM/bias/Adam/Assign^rnn/LSTM/bias/Adam_1/Assign^rnn/LSTM/bias/Assign^rnn/LSTM/kernel/Adam/Assign^rnn/LSTM/kernel/Adam_1/Assign^rnn/LSTM/kernel/Assign^rnn/LSTM_1/bias/Adam/Assign^rnn/LSTM_1/bias/Adam_1/Assign^rnn/LSTM_1/bias/Assign^rnn/LSTM_1/kernel/Adam/Assign ^rnn/LSTM_1/kernel/Adam_1/Assign^rnn/LSTM_1/kernel/Assign^rnn/LSTM_2/bias/Adam/Assign^rnn/LSTM_2/bias/Adam_1/Assign^rnn/LSTM_2/bias/Assign^rnn/LSTM_2/kernel/Adam/Assign ^rnn/LSTM_2/kernel/Adam_1/Assign^rnn/LSTM_2/kernel/Assign^rnn/LSTM_3/bias/Adam/Assign^rnn/LSTM_3/bias/Adam_1/Assign^rnn/LSTM_3/bias/Assign^rnn/LSTM_3/kernel/Adam/Assign ^rnn/LSTM_3/kernel/Adam_1/Assign^rnn/LSTM_3/kernel/Assign^rnn/LSTM_4/bias/Adam/Assign^rnn/LSTM_4/bias/Adam_1/Assign^rnn/LSTM_4/bias/Assign^rnn/LSTM_4/kernel/Adam/Assign ^rnn/LSTM_4/kernel/Adam_1/Assign^rnn/LSTM_4/kernel/Assign^rnn/LSTM_5/bias/Adam/Assign^rnn/LSTM_5/bias/Adam_1/Assign^rnn/LSTM_5/bias/Assign^rnn/LSTM_5/kernel/Adam/Assign ^rnn/LSTM_5/kernel/Adam_1/Assign^rnn/LSTM_5/kernel/Assign^rnn/LSTM_6/bias/Adam/Assign^rnn/LSTM_6/bias/Adam_1/Assign^rnn/LSTM_6/bias/Assign^rnn/LSTM_6/kernel/Adam/Assign ^rnn/LSTM_6/kernel/Adam_1/Assign^rnn/LSTM_6/kernel/Assign"