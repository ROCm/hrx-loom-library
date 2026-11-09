// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

[[loom::symbol("qwen3_embedding.attention_query"), loom::inline]]
void attention_query(Bytes storage, WeightStream input,
                     FeatureStream probabilities, FeatureStream results,
                     Bytes output, loomt::offset base, loomt::offset stride,
                     bool emit, loomt::index wave, loomt::index first_panel,
                     loomt::index key_begin, loomt::index key_end,
                     loomt::offset checkpoint_base);
[[loom::symbol("qwen3_embedding.attention_value"), loom::inline]]
void attention_value(Bytes storage, Bytes query_storage, WeightStream input,
                     FeatureStream probabilities, FeatureStream results,
                     Bytes output, loomt::offset base, loomt::offset stride,
                     bool emit, loomt::index wave, loomt::index first_panel,
                     loomt::index key_begin, loomt::index key_end,
                     loomt::offset checkpoint_base);

[[loom::inline]] static void fill_records(
    Bytes state, Bytes metadata, WeightStream source, loomt::index count,
    loomt::index wave, loomt::index key_begin, loomt::index fresh,
    loomt::index wave_count, loomt::index first_panel, loomt::index column) {
  const loomt::offset stride{3989504};
  auto column_offset = loom::index::scale(column, loomt::offset{32768});
  for (loomt::index record{0}; record < count;
       record = record + loomt::index{1}) {
    loomt::index absolute = key_begin + record;
    auto physical_panel = first_panel + absolute / loomt::index{2};
    loom::assume(loom::predicate::range(physical_panel, 0, 511));
    loomt::offset panel = loom::index::scale(physical_panel, stride);
    auto metadata_panel =
        loom::index::scale(physical_panel, loomt::offset{8192});
    loomt::offset half =
        loom::index::scale(absolute % loomt::index{2}, loomt::offset{16384});
    auto key = loom::channel::reserve(source);
    auto value = loom::channel::reserve(source);
    if (record >= fresh && absolute < wave_count / loomt::index{2}) {
      {
        auto destination = loom::view::bitcast<
            loomt::view<loomt::shape<16, 512>, std::bfloat16_t>>(
            loom::view::subview<4096>(key.payload, {0}, {}));
        auto external = loom::buffer::view<16, 512>(
            loom::buffer::cast<const std::bfloat16_t>(state),
            loomt::offset{8192} + column_offset + panel + half, {});
        auto copy = loom::kernel::async::copy<
            loom::kernel::async::direction::global_to_workgroup,
            loom::cache::scope::workgroup, loom::cache::temporal::regular>(
            external, destination);
        loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
        {
          auto ids = loom::buffer::view<32>(
              loom::buffer::cast<const unsigned>(metadata),
              metadata_panel + loomt::offset{4} +
                  loom::index::scale(absolute % loomt::index{2},
                                     loomt::offset{4096}),
              {}, loom::encoding::layout::strided(loomt::index{16}));
          auto ids_destination =
              loom::view::subview<32>(key.payload, {4096}, {});
          auto ids_copy = loom::kernel::async::copy<
              loom::kernel::async::direction::global_to_workgroup,
              loom::cache::scope::workgroup, loom::cache::temporal::regular>(
              ids, ids_destination);
          loom::kernel::async::wait<0>(loom::kernel::async::group(ids_copy));
        }
        {
          auto ids = loom::buffer::view<32>(
              loom::buffer::cast<const unsigned>(metadata),
              metadata_panel + loomt::offset{2052} +
                  loom::index::scale(absolute % loomt::index{2},
                                     loomt::offset{4096}),
              {}, loom::encoding::layout::strided(loomt::index{16}));
          auto ids_destination =
              loom::view::subview<32>(key.payload, {4128}, {});
          auto ids_copy = loom::kernel::async::copy<
              loom::kernel::async::direction::global_to_workgroup,
              loom::cache::scope::workgroup, loom::cache::temporal::regular>(
              ids, ids_destination);
          loom::kernel::async::wait<0>(loom::kernel::async::group(ids_copy));
        }
      }
      {
        auto destination = loom::view::bitcast<
            loomt::view<loomt::shape<16, 512>, std::bfloat16_t>>(
            loom::view::subview<4096>(value.payload, {0}, {}));
        auto external = loom::buffer::view<16, 512>(
            loom::buffer::cast<const std::bfloat16_t>(state),
            loomt::offset{270336} + column_offset + panel +
                loom::index::scale(absolute % loomt::index{2},
                                   loomt::offset{1024}),
            {},
            loom::encoding::layout::strided(loomt::index{1024},
                                            loomt::index{1}));
        auto copy = loom::kernel::async::copy<
            loom::kernel::async::direction::global_to_workgroup,
            loom::cache::scope::workgroup, loom::cache::temporal::regular>(
            external, destination);
        loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
      }
    }
    loom::channel::publish(key.write);
    loom::channel::publish(value.write);
  }
  auto physical_panel = first_panel + wave / loomt::index{4};
  loom::assume(loom::predicate::range(physical_panel, 0, 511));
  loomt::offset panel = loom::index::scale(physical_panel, stride);
  auto metadata_panel = loom::index::scale(physical_panel, loomt::offset{8192});
  loomt::offset half =
      loom::index::scale(wave % loomt::index{4}, loomt::offset{8192});
  for (int head = 0; head < 2; ++head) {
    loomt::offset head_offset =
        loom::index::scale(loom::index::cast(head), loomt::offset{262144});
    auto query = loom::channel::reserve(source);
    auto context = loom::channel::reserve(source);
    auto destination = loom::view::bitcast<
        loomt::view<loomt::shape<16, 256>, std::bfloat16_t>>(
        loom::view::subview<2048>(query.payload, {0}, {}));
    auto external = loom::buffer::view<16, 256>(
        loom::buffer::cast<const std::bfloat16_t>(state),
        loomt::offset{532480} + column_offset + head_offset + panel + half, {});
    auto copy = loom::kernel::async::copy<
        loom::kernel::async::direction::global_to_workgroup,
        loom::cache::scope::workgroup, loom::cache::temporal::regular>(
        external, destination);
    loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
    auto ids = loom::buffer::view<32>(
        loom::buffer::cast<const unsigned>(metadata),
        metadata_panel + loomt::offset{4} +
            loom::index::scale(wave % loomt::index{4}, loomt::offset{2048}),
        {}, loom::encoding::layout::strided(loomt::index{16}));
    auto ids_destination = loom::view::subview<32>(query.payload, {4096}, {});
    auto ids_copy = loom::kernel::async::copy<
        loom::kernel::async::direction::global_to_workgroup,
        loom::cache::scope::workgroup, loom::cache::temporal::regular>(
        ids, ids_destination);
    loom::kernel::async::wait<0>(loom::kernel::async::group(ids_copy));
    if (key_begin != loomt::index{0}) {
      {
        auto saved = loom::buffer::view<64>(
            loom::buffer::cast<const unsigned>(state),
            loomt::offset{2924544} +
                loom::index::scale(column, loomt::offset{1024}) + panel +
                loom::index::scale(loom::index::cast(head),
                                   loomt::offset{8192}) +
                loom::index::scale(wave % loomt::index{4}, loomt::offset{256}),
            {});
        auto saved_destination =
            loom::view::subview<64>(query.payload, {4160}, {});
        auto saved_copy = loom::kernel::async::copy<
            loom::kernel::async::direction::global_to_workgroup,
            loom::cache::scope::workgroup, loom::cache::temporal::regular>(
            saved, saved_destination);
        loom::kernel::async::wait<0>(loom::kernel::async::group(saved_copy));
      }
      {
        auto context_destination =
            loom::view::bitcast<loomt::view<loomt::shape<16, 256>, float>>(
                loom::view::subview<4096>(context.payload, {0}, {}));
        auto context_external = loom::buffer::view<16, 256>(
            loom::buffer::cast<const float>(state),
            loomt::offset{2940928} +
                loom::index::scale(column, loomt::offset{131072}) +
                loom::index::scale(loom::index::cast(head),
                                   loomt::offset{65536}) +
                panel + half + half,
            {});
        auto context_copy = loom::kernel::async::copy<
            loom::kernel::async::direction::global_to_workgroup,
            loom::cache::scope::workgroup, loom::cache::temporal::regular>(
            context_external, context_destination);
        loom::kernel::async::wait<0>(loom::kernel::async::group(context_copy));
      }
    }
    loom::channel::publish(query.write);
    loom::channel::publish(context.write);
  }
}
[[loom::pipeline]] static void attention_window_column(
    [[loom::launch]] Bytes state, Bytes metadata, Bytes cache, WeightStream r0,
    WeightStream r1, WeightStream r2, WeightStream r3, loomt::index wave,
    loomt::index key_begin, loomt::index fresh, loomt::index wave_count,
    loomt::index first_panel, Bytes s0, Bytes s1, Bytes s2, Bytes s3,
    FeatureStream p0, FeatureStream p2, FeatureStream o0, FeatureStream o2,
    loomt::index column, bool query_emits, loomt::index fill_row) {
  auto source = bind_weight_ingress<20>(cache);
  auto window_end = key_begin + loomt::index{8};
  auto causal_end = wave / loomt::index{2} + loomt::index{1};
  auto key_end = causal_end < window_end ? causal_end : window_end;
  [[loom::strand(loomt::indices{column, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    auto a0 = loom::channel::acquire(source);
    auto a1 = loom::channel::acquire(source);
    auto a2 = loom::channel::acquire(source);
    auto a3 = loom::channel::acquire(source);
    auto a4 = loom::channel::acquire(source);
    auto a5 = loom::channel::acquire(source);
    auto a6 = loom::channel::acquire(source);
    auto a7 = loom::channel::acquire(source);
    auto a8 = loom::channel::acquire(source);
    auto a9 = loom::channel::acquire(source);
    auto a10 = loom::channel::acquire(source);
    auto a11 = loom::channel::acquire(source);
    auto a12 = loom::channel::acquire(source);
    auto a13 = loom::channel::acquire(source);
    auto a14 = loom::channel::acquire(source);
    auto a15 = loom::channel::acquire(source);
    auto a16 = loom::channel::acquire(source);
    auto a17 = loom::channel::acquire(source);
    auto a18 = loom::channel::acquire(source);
    auto a19 = loom::channel::acquire(source);
    auto q0 = loom::channel::fanout<3>(a16.read);
    auto qwrite0_0 = loom::channel::reserve(r0);
    loom::channel::copy(q0.remaining.first, qwrite0_0.write);
    auto qwrite0_1 = loom::channel::reserve(r1);
    loom::channel::copy(q0.remaining.second, qwrite0_1.write);
    auto q1 = loom::channel::fanout<3>(a17.read);
    auto qwrite1_0 = loom::channel::reserve(r0);
    loom::channel::copy(q1.remaining.first, qwrite1_0.write);
    auto qwrite1_1 = loom::channel::reserve(r1);
    loom::channel::copy(q1.remaining.second, qwrite1_1.write);
    auto q2 = loom::channel::fanout<3>(a18.read);
    auto qwrite2_2 = loom::channel::reserve(r2);
    loom::channel::copy(q2.remaining.first, qwrite2_2.write);
    auto qwrite2_3 = loom::channel::reserve(r3);
    loom::channel::copy(q2.remaining.second, qwrite2_3.write);
    auto q3 = loom::channel::fanout<3>(a19.read);
    auto qwrite3_2 = loom::channel::reserve(r2);
    loom::channel::copy(q3.remaining.first, qwrite3_2.write);
    auto qwrite3_3 = loom::channel::reserve(r3);
    loom::channel::copy(q3.remaining.second, qwrite3_3.write);
    auto k0 = loom::channel::fanout<5>(a0.read);
    auto w0_0 = loom::channel::reserve(r0);
    loom::channel::copy(k0.remaining.first, w0_0.write);
    auto w0_1 = loom::channel::reserve(r1);
    loom::channel::copy(k0.remaining.remaining.first, w0_1.write);
    auto w0_2 = loom::channel::reserve(r2);
    loom::channel::copy(k0.remaining.remaining.remaining.first, w0_2.write);
    auto w0_3 = loom::channel::reserve(r3);
    loom::channel::copy(k0.remaining.remaining.remaining.second, w0_3.write);
    auto k1 = loom::channel::fanout<5>(a1.read);
    auto w1_0 = loom::channel::reserve(r0);
    loom::channel::copy(k1.remaining.first, w1_0.write);
    auto w1_1 = loom::channel::reserve(r1);
    loom::channel::copy(k1.remaining.remaining.first, w1_1.write);
    auto w1_2 = loom::channel::reserve(r2);
    loom::channel::copy(k1.remaining.remaining.remaining.first, w1_2.write);
    auto w1_3 = loom::channel::reserve(r3);
    loom::channel::copy(k1.remaining.remaining.remaining.second, w1_3.write);
    auto k2 = loom::channel::fanout<5>(a2.read);
    auto w2_0 = loom::channel::reserve(r0);
    loom::channel::copy(k2.remaining.first, w2_0.write);
    auto w2_1 = loom::channel::reserve(r1);
    loom::channel::copy(k2.remaining.remaining.first, w2_1.write);
    auto w2_2 = loom::channel::reserve(r2);
    loom::channel::copy(k2.remaining.remaining.remaining.first, w2_2.write);
    auto w2_3 = loom::channel::reserve(r3);
    loom::channel::copy(k2.remaining.remaining.remaining.second, w2_3.write);
    auto k3 = loom::channel::fanout<5>(a3.read);
    auto w3_0 = loom::channel::reserve(r0);
    loom::channel::copy(k3.remaining.first, w3_0.write);
    auto w3_1 = loom::channel::reserve(r1);
    loom::channel::copy(k3.remaining.remaining.first, w3_1.write);
    auto w3_2 = loom::channel::reserve(r2);
    loom::channel::copy(k3.remaining.remaining.remaining.first, w3_2.write);
    auto w3_3 = loom::channel::reserve(r3);
    loom::channel::copy(k3.remaining.remaining.remaining.second, w3_3.write);
    auto k4 = loom::channel::fanout<5>(a4.read);
    auto w4_0 = loom::channel::reserve(r0);
    loom::channel::copy(k4.remaining.first, w4_0.write);
    auto w4_1 = loom::channel::reserve(r1);
    loom::channel::copy(k4.remaining.remaining.first, w4_1.write);
    auto w4_2 = loom::channel::reserve(r2);
    loom::channel::copy(k4.remaining.remaining.remaining.first, w4_2.write);
    auto w4_3 = loom::channel::reserve(r3);
    loom::channel::copy(k4.remaining.remaining.remaining.second, w4_3.write);
    auto k5 = loom::channel::fanout<5>(a5.read);
    auto w5_0 = loom::channel::reserve(r0);
    loom::channel::copy(k5.remaining.first, w5_0.write);
    auto w5_1 = loom::channel::reserve(r1);
    loom::channel::copy(k5.remaining.remaining.first, w5_1.write);
    auto w5_2 = loom::channel::reserve(r2);
    loom::channel::copy(k5.remaining.remaining.remaining.first, w5_2.write);
    auto w5_3 = loom::channel::reserve(r3);
    loom::channel::copy(k5.remaining.remaining.remaining.second, w5_3.write);
    auto k6 = loom::channel::fanout<5>(a6.read);
    auto w6_0 = loom::channel::reserve(r0);
    loom::channel::copy(k6.remaining.first, w6_0.write);
    auto w6_1 = loom::channel::reserve(r1);
    loom::channel::copy(k6.remaining.remaining.first, w6_1.write);
    auto w6_2 = loom::channel::reserve(r2);
    loom::channel::copy(k6.remaining.remaining.remaining.first, w6_2.write);
    auto w6_3 = loom::channel::reserve(r3);
    loom::channel::copy(k6.remaining.remaining.remaining.second, w6_3.write);
    auto k7 = loom::channel::fanout<5>(a7.read);
    auto w7_0 = loom::channel::reserve(r0);
    loom::channel::copy(k7.remaining.first, w7_0.write);
    auto w7_1 = loom::channel::reserve(r1);
    loom::channel::copy(k7.remaining.remaining.first, w7_1.write);
    auto w7_2 = loom::channel::reserve(r2);
    loom::channel::copy(k7.remaining.remaining.remaining.first, w7_2.write);
    auto w7_3 = loom::channel::reserve(r3);
    loom::channel::copy(k7.remaining.remaining.remaining.second, w7_3.write);
    auto k8 = loom::channel::fanout<5>(a8.read);
    auto w8_0 = loom::channel::reserve(r0);
    loom::channel::copy(k8.remaining.first, w8_0.write);
    auto w8_1 = loom::channel::reserve(r1);
    loom::channel::copy(k8.remaining.remaining.first, w8_1.write);
    auto w8_2 = loom::channel::reserve(r2);
    loom::channel::copy(k8.remaining.remaining.remaining.first, w8_2.write);
    auto w8_3 = loom::channel::reserve(r3);
    loom::channel::copy(k8.remaining.remaining.remaining.second, w8_3.write);
    auto k9 = loom::channel::fanout<5>(a9.read);
    auto w9_0 = loom::channel::reserve(r0);
    loom::channel::copy(k9.remaining.first, w9_0.write);
    auto w9_1 = loom::channel::reserve(r1);
    loom::channel::copy(k9.remaining.remaining.first, w9_1.write);
    auto w9_2 = loom::channel::reserve(r2);
    loom::channel::copy(k9.remaining.remaining.remaining.first, w9_2.write);
    auto w9_3 = loom::channel::reserve(r3);
    loom::channel::copy(k9.remaining.remaining.remaining.second, w9_3.write);
    auto k10 = loom::channel::fanout<5>(a10.read);
    auto w10_0 = loom::channel::reserve(r0);
    loom::channel::copy(k10.remaining.first, w10_0.write);
    auto w10_1 = loom::channel::reserve(r1);
    loom::channel::copy(k10.remaining.remaining.first, w10_1.write);
    auto w10_2 = loom::channel::reserve(r2);
    loom::channel::copy(k10.remaining.remaining.remaining.first, w10_2.write);
    auto w10_3 = loom::channel::reserve(r3);
    loom::channel::copy(k10.remaining.remaining.remaining.second, w10_3.write);
    auto k11 = loom::channel::fanout<5>(a11.read);
    auto w11_0 = loom::channel::reserve(r0);
    loom::channel::copy(k11.remaining.first, w11_0.write);
    auto w11_1 = loom::channel::reserve(r1);
    loom::channel::copy(k11.remaining.remaining.first, w11_1.write);
    auto w11_2 = loom::channel::reserve(r2);
    loom::channel::copy(k11.remaining.remaining.remaining.first, w11_2.write);
    auto w11_3 = loom::channel::reserve(r3);
    loom::channel::copy(k11.remaining.remaining.remaining.second, w11_3.write);
    auto k12 = loom::channel::fanout<5>(a12.read);
    auto w12_0 = loom::channel::reserve(r0);
    loom::channel::copy(k12.remaining.first, w12_0.write);
    auto w12_1 = loom::channel::reserve(r1);
    loom::channel::copy(k12.remaining.remaining.first, w12_1.write);
    auto w12_2 = loom::channel::reserve(r2);
    loom::channel::copy(k12.remaining.remaining.remaining.first, w12_2.write);
    auto w12_3 = loom::channel::reserve(r3);
    loom::channel::copy(k12.remaining.remaining.remaining.second, w12_3.write);
    auto k13 = loom::channel::fanout<5>(a13.read);
    auto w13_0 = loom::channel::reserve(r0);
    loom::channel::copy(k13.remaining.first, w13_0.write);
    auto w13_1 = loom::channel::reserve(r1);
    loom::channel::copy(k13.remaining.remaining.first, w13_1.write);
    auto w13_2 = loom::channel::reserve(r2);
    loom::channel::copy(k13.remaining.remaining.remaining.first, w13_2.write);
    auto w13_3 = loom::channel::reserve(r3);
    loom::channel::copy(k13.remaining.remaining.remaining.second, w13_3.write);
    auto k14 = loom::channel::fanout<5>(a14.read);
    auto w14_0 = loom::channel::reserve(r0);
    loom::channel::copy(k14.remaining.first, w14_0.write);
    auto w14_1 = loom::channel::reserve(r1);
    loom::channel::copy(k14.remaining.remaining.first, w14_1.write);
    auto w14_2 = loom::channel::reserve(r2);
    loom::channel::copy(k14.remaining.remaining.remaining.first, w14_2.write);
    auto w14_3 = loom::channel::reserve(r3);
    loom::channel::copy(k14.remaining.remaining.remaining.second, w14_3.write);
    auto k15 = loom::channel::fanout<5>(a15.read);
    auto w15_0 = loom::channel::reserve(r0);
    loom::channel::copy(k15.remaining.first, w15_0.write);
    auto w15_1 = loom::channel::reserve(r1);
    loom::channel::copy(k15.remaining.remaining.first, w15_1.write);
    auto w15_2 = loom::channel::reserve(r2);
    loom::channel::copy(k15.remaining.remaining.remaining.first, w15_2.write);
    auto w15_3 = loom::channel::reserve(r3);
    loom::channel::copy(k15.remaining.remaining.remaining.second, w15_3.write);
    loom::channel::release(k0.first);
    loom::channel::release(k1.first);
    loom::channel::release(k2.first);
    loom::channel::release(k3.first);
    loom::channel::release(k4.first);
    loom::channel::release(k5.first);
    loom::channel::release(k6.first);
    loom::channel::release(k7.first);
    loom::channel::release(k8.first);
    loom::channel::release(k9.first);
    loom::channel::release(k10.first);
    loom::channel::release(k11.first);
    loom::channel::release(k12.first);
    loom::channel::release(k13.first);
    loom::channel::release(k14.first);
    loom::channel::release(k15.first);
    loom::channel::release(q0.first);
    loom::channel::release(q1.first);
    loom::channel::release(q2.first);
    loom::channel::release(q3.first);
  }
  [[loom::strand(loomt::indices{column, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (fill_row == loomt::index{2}) {
      fill_records(state, metadata, source, loomt::index{8}, wave, key_begin,
                   fresh, wave_count, first_panel, column);
    }
    attention_query(s0, r0, p0, o0, state,
                    loomt::offset{1318912} +
                        loom::index::scale(column, loomt::offset{65536}),
                    loomt::offset{3989504}, query_emits, wave, first_panel,
                    key_begin, key_end,
                    loomt::offset{2924544} +
                        loom::index::scale(column, loomt::offset{1024}));
  }
  [[loom::strand(loomt::indices{column, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (fill_row == loomt::index{3}) {
      fill_records(state, metadata, source, loomt::index{8}, wave, key_begin,
                   fresh, wave_count, first_panel, column);
    }
    attention_value(s1, s0, r1, p0, o0, state,
                    loomt::offset{1318912} +
                        loom::index::scale(column, loomt::offset{65536}),
                    loomt::offset{3989504}, !query_emits, wave, first_panel,
                    key_begin, key_end,
                    loomt::offset{2924544} +
                        loom::index::scale(column, loomt::offset{1024}));
  }
  [[loom::strand(loomt::indices{column, loomt::index{4}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (fill_row == loomt::index{4}) {
      fill_records(state, metadata, source, loomt::index{8}, wave, key_begin,
                   fresh, wave_count, first_panel, column);
    }
    attention_query(s2, r2, p2, o2, state,
                    loomt::offset{1351680} +
                        loom::index::scale(column, loomt::offset{65536}),
                    loomt::offset{3989504}, query_emits, wave, first_panel,
                    key_begin, key_end,
                    loomt::offset{2932736} +
                        loom::index::scale(column, loomt::offset{1024}));
  }
  [[loom::strand(loomt::indices{column, loomt::index{5}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    if (fill_row == loomt::index{5}) {
      fill_records(state, metadata, source, loomt::index{8}, wave, key_begin,
                   fresh, wave_count, first_panel, column);
    }
    attention_value(s3, s2, r3, p2, o2, state,
                    loomt::offset{1351680} +
                        loom::index::scale(column, loomt::offset{65536}),
                    loomt::offset{3989504}, !query_emits, wave, first_panel,
                    key_begin, key_end,
                    loomt::offset{2932736} +
                        loom::index::scale(column, loomt::offset{1024}));
  }
}
