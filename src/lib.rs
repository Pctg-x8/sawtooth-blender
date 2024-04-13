use bedrock as br;
use br::{GraphicsPipelineBuilder, Image, ImageChild, SubmissionBatch};
use peridot::{EngineEvents, FeatureRequests, NativeLinker, SpecConstantStorage};
use peridot_command_object::{
    BeginRenderPass, BufferUsage, EndRenderPass, GraphicsCommand, GraphicsCommandCombiner,
    RangedBuffer, StandardMesh,
};
use peridot_memory_manager::MemoryManager;
use peridot_vertex_processing_pack::{PvpContainer, PvpShaderModules};
use rand::distributions::Distribution;

const GENERATE_SAMPLES: usize = 512;

#[derive(SpecConstantStorage)]
pub struct GraphShaderVertexSpecConstants {
    pub generate_samples: u32,
}

pub struct Game<NL: NativeLinker> {
    main_render_pass: br::RenderPassObject<peridot::DeviceObject>,
    framebuffers: Vec<br::FramebufferObject<'static, peridot::DeviceObject>>,
    graph_mesh: StandardMesh<peridot_memory_manager::Buffer>,
    graph_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        br::PipelineLayoutObject<peridot::DeviceObject>,
    >,
    render_cb: peridot::CommandBundle<peridot::DeviceObject>,
    _ph: std::marker::PhantomData<*const NL>,
}
impl<NL: NativeLinker> FeatureRequests for Game<NL> {}
impl<NL: NativeLinker> EngineEvents<NL> for Game<NL> {
    fn init(e: &mut peridot::Engine<NL>) -> Self {
        let main_render_pass = br::RenderPassBuilder::new()
            .add_attachment(
                e.back_buffer_attachment_desc()
                    .color_memory_op(br::LoadOp::Clear, br::StoreOp::Store),
            )
            .add_subpass(br::SubpassDescription::new().add_color_output(
                0,
                br::ImageLayout::ColorAttachmentOpt,
                None,
            ))
            .add_dependency(br::vk::VkSubpassDependency {
                srcSubpass: br::vk::VK_SUBPASS_EXTERNAL,
                dstSubpass: 0,
                srcStageMask: br::PipelineStageFlags::ALL_COMMANDS.0,
                dstStageMask: br::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.0,
                srcAccessMask: 0,
                dstAccessMask: br::AccessFlags::COLOR_ATTACHMENT.write,
                dependencyFlags: br::vk::VK_DEPENDENCY_BY_REGION_BIT,
            })
            .create(e.graphics_device().clone())
            .expect("Failed to create main renderpass");
        let framebuffers = e
            .iter_back_buffers()
            .map(|bb| {
                br::FramebufferBuilder::new_with_attachment(&main_render_pass, bb.clone()).create()
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create framebuffers");

        let mut memory_manager = MemoryManager::new(e.graphics());

        let mut generated = [0.0f32; GENERATE_SAMPLES];
        let f = 1.0f32;
        let mut m = 1.0f32;
        let mut ph = 0.0f32;
        while f * m <= (GENERATE_SAMPLES / 2) as f32 {
            ph = rand::distributions::Uniform::new(0.0f32, 1.0f32).sample(&mut rand::thread_rng());
            for (x, d) in generated.iter_mut().enumerate() {
                *d += (f * m * std::f32::consts::TAU * x as f32 / GENERATE_SAMPLES as f32
                    + std::f32::consts::TAU * ph)
                    .sin()
                    / m;
            }

            m += 1.0;
        }
        for d in generated.iter_mut() {
            *d *= 2.0 / std::f32::consts::PI;
        }

        let buffer = memory_manager
            .allocate_device_local_buffer(
                e.graphics(),
                br::BufferDesc::new(
                    GENERATE_SAMPLES * 4,
                    br::BufferUsage::VERTEX_BUFFER.transfer_dest(),
                ),
            )
            .expect("Failed to allocate vertex buffer");
        let mut stg_buffer = memory_manager
            .allocate_upload_buffer(
                e.graphics(),
                br::BufferDesc::new(GENERATE_SAMPLES * 4, br::BufferUsage::TRANSFER_SRC),
            )
            .expect("Failed to allocate uplaod buffer");
        stg_buffer
            .write_content(generated)
            .expect("Failed to write upload content");

        let vertex_buffer = RangedBuffer::from(buffer);
        let vertex_upload_buffer = RangedBuffer::from(stg_buffer);
        e.submit_commands(|mut rec| {
            let [vbuf_enter_barrier, vbuf_leave_barrier] = vertex_buffer.make_ref().usage_barrier3(
                BufferUsage::UNUSED,
                BufferUsage::TRANSFER_DST,
                BufferUsage::VERTEX_BUFFER,
            );
            let vbuf_upload_enter_barrier = vertex_upload_buffer
                .make_ref()
                .usage_barrier(BufferUsage::HOST_RW, BufferUsage::TRANSFER_SRC);

            vertex_buffer
                .byref_mirror_from(&vertex_upload_buffer)
                .between(
                    [vbuf_enter_barrier, vbuf_upload_enter_barrier],
                    [vbuf_leave_barrier],
                )
                .execute(&mut rec.as_dyn_ref());

            rec
        })
        .expect("Failed to execute initialize commands");
        let graph_mesh = StandardMesh {
            vertex_buffers: vec![vertex_buffer],
            vertex_count: GENERATE_SAMPLES as _,
        };

        let scissor = [e
            .back_buffer(0)
            .expect("no backbuffers?")
            .image()
            .size()
            .wh()
            .into_rect(br::vk::VkOffset2D::ZERO)];
        let viewport = [scissor[0].make_viewport(0.0..1.0)];

        let graph_shader = e
            .load::<PvpContainer>("shaders.graph")
            .expect("Failed to load graph shader");
        let mut graph_shader = PvpShaderModules::new(e.graphics_device(), graph_shader)
            .expect("Failed to instantiate shader");
        let graph_shader_vsh_constants = GraphShaderVertexSpecConstants {
            generate_samples: GENERATE_SAMPLES as _,
        };
        let (a, b) = graph_shader_vsh_constants.as_pair();
        graph_shader.set_vertex_spec_constants(a, b);

        let graph_pipeline_layout = br::PipelineLayoutBuilder::new(vec![], vec![])
            .create(e.graphics_device().clone())
            .expect("Failed to create graph pipeline layout");
        let mut pb = br::NonDerivedGraphicsPipelineBuilder::new(
            &graph_pipeline_layout,
            (&main_render_pass, 0),
            graph_shader.generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP),
        );
        pb.add_attachment_blend(br::AttachmentColorBlendState::noblend())
            .multisample_state(Some(br::MultisampleState::new()))
            .viewport_scissors(
                br::DynamicArrayState::Static(&viewport),
                br::DynamicArrayState::Static(&scissor),
            );
        let graph_pipeline = pb
            .create(
                e.graphics_device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create graph pieline");
        let graph_pipeline =
            peridot::LayoutedPipeline::combine(graph_pipeline, graph_pipeline_layout);

        let mut render_cb = peridot::CommandBundle::new(
            e.graphics(),
            peridot::CBSubmissionType::Graphics,
            framebuffers.len(),
        )
        .expect("Failed to create render command bundle");
        let r_main_pass = graph_mesh.ref_draw(1).after_of(&graph_pipeline);
        for (n, fb) in framebuffers.iter().enumerate() {
            let mut cb_sync = render_cb.synchronized_nth(n);
            let mut rec = cb_sync.begin().expect("Failed to begin recording commands");
            let brp = BeginRenderPass::for_entire_framebuffer(&main_render_pass, fb)
                .with_clear_values(vec![br::ClearValue::color_f32([0.0, 0.0, 0.0, 1.0])]);

            (&r_main_pass)
                .between(brp, EndRenderPass)
                .execute_and_finish(rec.as_dyn_ref())
                .expect("Failed to record commands");
        }

        Game {
            main_render_pass,
            framebuffers,
            graph_mesh,
            graph_pipeline,
            render_cb,
            _ph: std::marker::PhantomData,
        }
    }

    fn update(
        &mut self,
        e: &mut peridot::Engine<NL>,
        on_back_buffer_of: u32,
        _delta_time: std::time::Duration,
    ) {
        e.do_render(
            on_back_buffer_of,
            None::<br::EmptySubmissionBatch>,
            br::EmptySubmissionBatch.with_command_buffers(
                &self.render_cb[on_back_buffer_of as usize..=on_back_buffer_of as usize],
            ),
        )
        .expect("Failed to submit rendering");
    }
}
