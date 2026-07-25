// models/schema.js — ModelSpec 校验器
// 契约：window.validateSpec(spec)，缺字段 console.warn 并补默认值，返回补全后的 spec。
// 基础组件类型（契约枚举）：tokenizer/embedding/kda/mla/router/experts/attnres/norm/output/softmax
// 注：glm52 的 attn、minimax3 的 lightning-attn、deepseekv4 的 nsa-attn 为约定的占位扩展，
// 不在此枚举内属预期行为，本校验器不对 type 枚举做告警（渲染回退由 engine 负责）。
(function () {
  'use strict';

  // 基础组件类型表，导出给 engine/content 做协调参考
  window.MODEL_COMPONENT_TYPES = Object.freeze([
    'tokenizer', 'embedding', 'kda', 'mla', 'router',
    'experts', 'attnres', 'norm', 'output', 'softmax'
  ]);

  var SCALAR_DEFAULTS = {
    id: 'unknown',
    name: '未命名模型',
    sourceTag: 'estimated',
    context: 0,
    blockCount: 1,
    d_model: 0,
    quant: null // null = 无量化，属合法值；仅 undefined 时告警
  };

  function warn(id, msg) {
    console.warn('[validateSpec] ' + id + ': ' + msg);
  }

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  window.validateSpec = function (spec) {
    if (!spec || typeof spec !== 'object') {
      console.warn('[validateSpec] spec 不是对象，已用空默认 spec 替换');
      spec = {};
    }
    var id = (typeof spec.id === 'string' && spec.id) || SCALAR_DEFAULTS.id;

    // 标量字段：undefined 才补默认值（quant/moe 显式 null 合法）
    Object.keys(SCALAR_DEFAULTS).forEach(function (k) {
      if (spec[k] === undefined) {
        warn(id, '缺字段 "' + k + '"，补默认值 ' + JSON.stringify(SCALAR_DEFAULTS[k]));
        spec[k] = SCALAR_DEFAULTS[k];
      }
    });
    if (spec.sourceTag !== 'official' && spec.sourceTag !== 'estimated') {
      warn(id, 'sourceTag 只能是 official/estimated，已改为 estimated');
      spec.sourceTag = 'estimated';
    }

    // params {total, active}
    if (!spec.params || typeof spec.params !== 'object') {
      warn(id, '缺字段 "params"，补默认值 {total:"未知", active:"未知"}');
      spec.params = { total: '未知', active: '未知' };
    } else {
      if (spec.params.total === undefined) {
        warn(id, '缺字段 "params.total"，补默认值 "未知"');
        spec.params.total = '未知';
      }
      if (spec.params.active === undefined) {
        warn(id, '缺字段 "params.active"，补默认值 "未知"');
        spec.params.active = '未知';
      }
    }

    // layers [{type,...}]，自底向上；字符串项自动包装为 {type}
    if (!Array.isArray(spec.layers)) {
      warn(id, '缺字段 "layers"（或不是数组），补默认值 []');
      spec.layers = [];
    } else {
      spec.layers = spec.layers.reduce(function (acc, l) {
        if (typeof l === 'string') { acc.push({ type: l }); return acc; }
        if (!l || typeof l.type !== 'string' || !l.type) {
          warn(id, 'layers 含缺 type 的项，已丢弃');
          return acc;
        }
        acc.push(l);
        return acc;
      }, []);
    }

    // moe {experts, active, shared}；显式 null = 非 MoE，合法
    if (spec.moe === undefined) {
      warn(id, '缺字段 "moe"，补默认值 null（按非 MoE 处理）');
      spec.moe = null;
    } else if (spec.moe !== null) {
      var moeDefaults = { experts: 1, active: 1, shared: 0 };
      Object.keys(moeDefaults).forEach(function (k) {
        if (!isNum(spec.moe[k])) {
          warn(id, 'moe.' + k + ' 缺失或非数字，补默认值 ' + moeDefaults[k]);
          spec.moe[k] = moeDefaults[k];
        }
      });
    }

    // flow [[x,y,z],...]；坏点丢弃并告警
    if (!Array.isArray(spec.flow)) {
      warn(id, '缺字段 "flow"（或不是数组），补默认值 []');
      spec.flow = [];
    } else {
      spec.flow = spec.flow.reduce(function (acc, p) {
        if (!Array.isArray(p) || p.length !== 3) {
          warn(id, 'flow 含非 [x,y,z] 三元组的点，已丢弃');
          return acc;
        }
        var pt = [Number(p[0]), Number(p[1]), Number(p[2])];
        if (pt.some(function (v) { return !isFinite(v); })) {
          warn(id, 'flow 含非数字坐标点，已丢弃');
          return acc;
        }
        acc.push(pt);
        return acc;
      }, []);
    }

    return spec;
  };
})();
